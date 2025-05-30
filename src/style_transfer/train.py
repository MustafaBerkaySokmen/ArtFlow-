# train.py
"""
Training and fine-tuning logic for style transfer adapters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor
from .utils import tidy_cuda, fix_model_dtypes


def train_adapters(model, train_ds, val_ds, epochs=5, batch_size=4, lr=5e-5, style_alpha=0.05, grad_clip=0.5, device=None, ckpt_dir="checkpoints"):
    """
    Fine-tune only the adapters for a few epochs.
    """
    import os
    import itertools
    os.makedirs(ckpt_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fix_model_dtypes(model)
    for p in itertools.chain(model.unet.parameters(),
                             model.pipe.text_encoder.parameters(),
                             model.pipe.vae.parameters(),
                             model.cond.parameters()):
        p.requires_grad_(False)
    model.cond = model.cond.to(device, torch.float32).eval()
    for i, adpt in enumerate(model.adapters):
        model.adapters[i] = adpt.to(device, torch.float32)
    model.unet.set_attn_processor({n: AttnProcessor() for n in model.unet.attn_processors})
    with torch.no_grad():
        tok_ids  = model.pipe.tokenizer("", return_tensors="pt").input_ids.to(device)
        base_emb = model.pipe.text_encoder(tok_ids)[0].to(torch.float16).detach()
    optim  = torch.optim.Adam(model.adapters.parameters(), lr=lr)
    scaler = GradScaler()
    sched  = DDPMScheduler(1000, beta_schedule="squaredcos_cap_v2")
    def _norm(x, eps=1e-3):
        return (x - x.mean()) / (x.std() + eps)
    for epoch in range(1, epochs + 1):
        model.adapters.train();   train_loss = 0.0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        for ct, st in train_loader:
            ct, st = ct.to(device), st.to(device);  B = ct.size(0)
            with torch.no_grad(), autocast(dtype=torch.float16):
                lat = model.pipe.vae.encode((ct*2-1)).latent_dist.sample()
                lat = lat * model.pipe.vae.config.scaling_factor
            t     = torch.randint(0, 1000, (B,), device=device)
            noise = torch.randn_like(lat)
            noisy = sched.add_noise(lat, noise, t).float()
            with torch.no_grad():
                s_map = model.cond(ct.float(), st.float())
            model._style = (style_alpha * _norm(s_map)).to(torch.float16)
            optim.zero_grad(set_to_none=True)
            with autocast(dtype=torch.float16):
                pred = model.unet(noisy.to(torch.float16), t,
                                  encoder_hidden_states=base_emb.repeat(B,1,1)).sample
                loss = F.mse_loss(pred.float(), noise)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.adapters.parameters(), grad_clip)
            scaler.step(optim); scaler.update()
            model._style = None
            train_loss  += loss.item()
            torch.cuda.empty_cache()
        train_loss /= len(train_loader)
        # Validation
        model.adapters.eval(); val_loss = 0.0
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        with torch.no_grad():
            for ct, st in val_loader:
                ct, st = ct.to(device), st.to(device); B = ct.size(0)
                with autocast(dtype=torch.float16):
                    lat = model.pipe.vae.encode((ct*2-1)).latent_dist.sample()
                    lat = lat * model.pipe.vae.config.scaling_factor
                t     = torch.randint(0, 1000, (B,), device=device)
                noise = torch.randn_like(lat)
                noisy = sched.add_noise(lat, noise, t).float()
                s_map = model.cond(ct.float(), st.float())
                model._style = (style_alpha * _norm(s_map)).to(torch.float16)
                with autocast(dtype=torch.float16):
                    pred = model.unet(noisy.to(torch.float16), t,
                                      encoder_hidden_states=base_emb.repeat(B,1,1)).sample
                    val_loss += F.mse_loss(pred.float(), noise).item()
                model._style = None
        val_loss /= len(val_loader)
        torch.save({"epoch": epoch,
                    "adapters": model.adapters.state_dict(),
                    "optim": optim.state_dict()},
                   f"{ckpt_dir}/epoch_{epoch}.pt")
        print(f"Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f}")
        import gc; gc.collect(); torch.cuda.empty_cache()
    print("🏁 training finished – adapters fine‑tuned.")
