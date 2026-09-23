"""Structured pruning helper based on the pruning strategy used in the notebook."""
import torch.nn as nn
import torch.nn.utils.prune as prune

def structured_prune_conv_layers(model, amount=0.1, dim=0, remove_reparam=False):
    """Apply Ln structured pruning to Conv2d weights."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=dim)
            if remove_reparam:
                prune.remove(module, "weight")
    return model
