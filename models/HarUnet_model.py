import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Residual Conv Block (stride only in first conv)
# ----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super().__init__()
        self.strides = strides
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0, bias=True)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        return out1 + out2

    def flops(self, H, W):
        # rough estimate; ignores activations/bias
        k3 = 3 * 3
        k1 = 1 * 1
        f1 = H * W * self.in_channel * self.out_channel * k3
        f2 = H * W * self.out_channel * self.out_channel * k3
        f3 = H * W * self.in_channel * self.out_channel * k1 if (self.in_channel != self.out_channel or self.strides != 1) else 0
        return f1 + f2 + f3


# ----------------------------
# Windowed Attention (noise-aware), conv QKV + relative bias
# ----------------------------
class WA(nn.Module):
    """
    Windowed 2D self-attention with conv QKV and relative position bias.
    Optional noise-aware tempering via `noise_map` (B,1,H,W) normalized ~[0,1].
    """
    def __init__(self, dim, window_size=16, num_heads=8, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.zeta_w = nn.Parameter(torch.ones(1))  # strength for noise tempering

        # Q, K, V via 1x1 convs
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.proj  = nn.Conv2d(dim, dim, kernel_size=1)

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute and register relative position index buffer
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, ws, ws)
        coords_flatten = coords.flatten(1)  # (2, L)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, L, L)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= (2 * self.window_size - 1)
        self.register_buffer(
            "relative_position_index",
            relative_coords.sum(-1).long(),
            persistent=False
        )

    def _partition_windows(self, x):
        # x: (B,C,H,W) → (Bn,C,ws,ws), plus shape info
        B, C, H, W = x.shape
        ws = self.window_size
        x = x.view(B, C, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, nH, nW, C, ws, ws)
        xw = x.view(-1, C, ws, ws)                    # (Bn, C, ws, ws)
        return xw, (B, C, H, W)

    def _reverse_windows(self, windows, shape_info):
        B, C, H, W = shape_info
        ws = self.window_size
        nH, nW = H // ws, W // ws
        x = windows.view(B, nH, nW, C, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W)
        return x

    def forward(self, x, noise_map=None):
        """
        x: (B,C,H,W)
        noise_map: (B,1,H,W) or None
        """
        B, C, H, W = x.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, "H and W must be divisible by window_size"

        # Partition
        xw, shape_info = self._partition_windows(x)  # (Bn,C,ws,ws)
        Bn = xw.size(0)
        L = ws * ws

        # QKV
        q = self.q_conv(xw)
        k = self.k_conv(xw)
        v = self.v_conv(xw)

        # reshape to (Bn, heads, L, head_dim)
        def to_heads(t):
            return t.view(Bn, self.num_heads, self.head_dim, L).permute(0, 1, 3, 2).contiguous()
        qh, kh, vh = to_heads(q), to_heads(k), to_heads(v)

        # attention logits
        qh = qh * self.scale
        attn = qh @ kh.transpose(-2, -1)  # (Bn, heads, L, L)

        # relative bias
        rel = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        rel = rel.view(L, L, self.num_heads).permute(2, 0, 1).contiguous()  # (heads, L, L)
        attn = attn + rel.unsqueeze(0)

        # Optional: noise-aware tempering (per window)
        if noise_map is not None:
            if noise_map.shape[-2:] != (H, W):
                noise_map = F.interpolate(noise_map, size=(H, W), mode="nearest")
            nm = noise_map.view(B, 1, H // ws, ws, W // ws, ws)
            nm = nm.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, 1, ws, ws)  # (Bn,1,ws,ws)
            nm_win = nm.mean(dim=(2, 3), keepdim=True)  # (Bn,1,1,1), assume ~[0,1]
            temp = 1.0 + self.zeta_w * (nm_win.squeeze(-1).squeeze(-1) - 0.5)  # (Bn,1)
            temp = temp.clamp(0.5, 1.5)
            attn = attn / temp.unsqueeze(1).unsqueeze(-1)

        # softmax & output
        attn = F.softmax(attn, dim=-1)
        out = attn @ vh  # (Bn, heads, L, head_dim)
        out = out.permute(0, 1, 3, 2).contiguous().view(Bn, C, ws, ws)  # (Bn,C,ws,ws)

        # merge & project
        out = self._reverse_windows(out, shape_info)  # (B,C,H,W)
        out = self.proj(out)
        return out


# ----------------------------
# Small helpers: LayerNorm2d + SE
# ----------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)
    def forward(self, x):  # (B,C,H,W)
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.mlp(self.pool(x))


# ----------------------------
# Hybrid Attention Bottleneck = PreNorm -> WA -> Residual -> PreNorm -> Conv-FFN -> SE -> Residual
# ----------------------------
class HAB(nn.Module):
    def __init__(self, dim, window_size=16, heads=8, mlp_ratio=2.0, se_reduction=8):
        super().__init__()
        assert dim % heads == 0, "bottleneck dim must be divisible by num_heads"
        self.norm1 = LayerNorm2d(dim)
        self.wa    = WA(dim, window_size=window_size, num_heads=heads, qkv_bias=True)

        self.norm2 = LayerNorm2d(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )
        self.se = SEBlock(dim, reduction=se_reduction)

    def forward(self, x, noise_map=None):
        residual = x
        y = self.wa(self.norm1(x))
        x = x + y
        z = self.ffn(self.norm2(x))
        z = self.se(z)
        x = x + z
        return x + residual


# ----------------------------

# ----------------------------
# HARU-Net with WA bottleneck + CBAM on skips
# ----------------------------
class HARU_net(nn.Module):
    def __init__(self, block=ConvBlock, dim=64, hab_heads=8, hab_ws=16, hab_mlp_ratio=2.0, hab_depth=6):
        super().__init__()
        self.dim = dim

        # Encoder
        self.ConvBlock1 = block(1, dim, strides=1)
        self.pool1 = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2, dim*2, kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(dim*4, dim*4, kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(dim*8, dim*8, kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.ConvBlock5 = block(dim*8, dim*16, strides=1)
        self.hag_block = nn.ModuleList([
            HAB(dim*16, window_size=hab_ws, heads=hab_heads, mlp_ratio=hab_mlp_ratio, se_reduction=8)
            for _ in range(hab_depth)
        ])

        # CBAM on skip tensors
        self.skip1 = HAB(dim, window_size=hab_ws, heads=hab_heads, mlp_ratio=hab_mlp_ratio, se_reduction=8)
        self.skip2 = HAB(dim*2, window_size=hab_ws, heads=hab_heads, mlp_ratio=hab_mlp_ratio, se_reduction=8)
        self.skip3 = HAB(dim*4, window_size=hab_ws, heads=hab_heads, mlp_ratio=hab_mlp_ratio, se_reduction=8)
        self.skip4 = HAB(dim*8, window_size=hab_ws, heads=hab_heads, mlp_ratio=hab_mlp_ratio, se_reduction=8)

        # Decoder
        self.upv6 = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*16, dim*8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*8, dim*4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*4, dim*2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim*2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 1, kernel_size=3, stride=1, padding=1)

        # quick sanity
        assert (dim*16) % hab_heads == 0, "bottleneck channels must be divisible by hab_heads"

    def forward(self, x, noise_map=None):
        # Encoder
        conv1 = self.ConvBlock1(x);  pool1 = self.pool1(conv1)
        conv2 = self.ConvBlock2(pool1); pool2 = self.pool2(conv2)
        conv3 = self.ConvBlock3(pool2); pool3 = self.pool3(conv3)
        conv4 = self.ConvBlock4(pool3); pool4 = self.pool4(conv4)

        # Bottleneck
        conv5 = self.ConvBlock5(pool4)
        conv5_ = conv5
        for hab in self.hag_block:
            # WA asserts window divisibility; make sure your input sizes cooperate
            conv5 = hab(conv5, noise_map=noise_map)

        conv5 = conv5 + conv5_
        # Decoder + CBAM-enhanced skips
        skip_connect4 = self.skip4(conv4)
        up6   = self.upv6(conv5)
        up6   = torch.cat([up6, skip_connect4], dim=1)
        conv6 = self.ConvBlock6(up6)

        skip_connect3 = self.skip3(conv3)
        up7   = self.upv7(conv6)
        up7   = torch.cat([up7, skip_connect3], dim=1)
        conv7 = self.ConvBlock7(up7)

        skip_connect2 = self.skip2(conv2)
        up8   = self.upv8(conv7)
        up8   = torch.cat([up8, skip_connect2], dim=1)
        conv8 = self.ConvBlock8(up8)

        skip_connect1 = self.skip1(conv1)
        up9   = self.upv9(conv8)
        up9   = torch.cat([up9, skip_connect1], dim=1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = x + conv10
        return out


