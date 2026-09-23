import numpy as np
import torch.nn.functional as F

def psnr(img1, img2, data_range=1.0):
    mse = F.mse_loss(img1, img2, reduction="mean").item()
    if mse == 0:
        return float("inf")
    return 20 * np.log10(data_range / np.sqrt(mse))

def batch_psnr(imgs1, imgs2, data_range=1.0):
    return float(np.mean([psnr(imgs1[i], imgs2[i], data_range) for i in range(imgs1.size(0))]))
