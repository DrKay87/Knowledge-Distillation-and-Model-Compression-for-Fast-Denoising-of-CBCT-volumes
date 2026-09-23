"""Utilities for quantization-aware training (QAT).

QAT backend/support depends on the PyTorch build and target hardware.
"""
import torch
import torch.ao.quantization as tq

def prepare_qat(model, backend="fbgemm"):
    torch.backends.quantized.engine = backend
    model.train()
    model.qconfig = tq.get_default_qat_qconfig(backend)
    return tq.prepare_qat(model, inplace=False)

def convert_qat_model(model):
    model = model.cpu().eval()
    return tq.convert(model, inplace=False)
