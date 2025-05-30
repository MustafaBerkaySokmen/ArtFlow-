# utils.py
"""
Utility functions for memory management and dtype handling.
"""
import gc
import torch

def tidy_cuda():
    """Free every tensor & empty the CUDA caching allocator."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def fix_model_dtypes(model, target_dtype=None):
    """
    Force every parameter (weight and bias) in UNet, adapters, text-encoder, and VAE to the same dtype.
    """
    if target_dtype is None:
        target_dtype = model.dtype
    def _cast_module(mod):
        if hasattr(mod, "weight") and mod.weight is not None:
            mod.weight.data = mod.weight.data.to(target_dtype)
        if hasattr(mod, "bias") and mod.bias is not None:
            mod.bias.data = mod.bias.data.to(target_dtype)
    for m in model.unet.modules():
        _cast_module(m)
    for i, adpt in enumerate(model.adapters):
        model.adapters[i] = adpt.to(model.device, target_dtype)
        _cast_module(model.adapters[i])
    for m in model.pipe.text_encoder.modules():
        _cast_module(m)
    for m in model.pipe.vae.modules():
        _cast_module(m)
    print(f"✅ All model parameters are now {target_dtype}")
    return model
