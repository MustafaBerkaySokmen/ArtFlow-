# model.py
"""
Core model and adapter logic for FFT-based style transfer with Stable Diffusion.
"""
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from fft_style_conditioner import FFTStyleConditioner

class StyleTransferDiffusion:
    """
    Loads Stable-Diffusion (SD-1.4), builds adapters, and injects FFT style features.
    Supports manual 'keep' (content/noise mix) and 'guidance' (style strength).
    """
    def __init__(self, model_id="CompVis/stable-diffusion-v1-4", device="cuda"):
        self.device = torch.device(device)
        self.dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        self.unet = self.pipe.unet
        self.cond = FFTStyleConditioner().to(self.device, dtype=torch.float32)
        self.sites = [
            (self.unet.down_blocks[2], self.unet.down_blocks[2].resnets[-1].out_channels),
            (self.unet.down_blocks[3], self.unet.down_blocks[3].resnets[-1].out_channels),
            (self.unet.mid_block,      self.unet.mid_block.resnets[-1].out_channels),
            (self.unet.up_blocks[0],   self.unet.up_blocks[0].resnets[-1].out_channels),
            (self.unet.up_blocks[1],   self.unet.up_blocks[1].resnets[-1].out_channels),
        ]
        self.adapters = torch.nn.ModuleList([
            torch.nn.Conv2d(512, ch, 1).to(self.device, self.dtype) for _, ch in self.sites
        ])
        for p in self.unet.parameters():
            p.requires_grad_(False)
        self._style = None
        self._bind_sites()

    def _proj(self, idx, h):
        s = self.adapters[idx](self._style)
        return F.interpolate(s, size=h.shape[-2:], mode="bilinear", align_corners=False)

    def _bind_sites(self):
        # down blocks 2 & 3
        for idx, (blk, _) in enumerate(self.sites[:2]):
            orig = blk.forward
            def wrap(self_blk, hidden_states, *a, _orig=orig, _idx=idx, **kw):
                h, skip = _orig(hidden_states, *a, **kw)
                if self._style is not None:
                    h = h + self._proj(_idx, h)
                return h, skip
            blk.forward = wrap.__get__(blk, blk.__class__)
        # mid block
        blk_mid, _ = self.sites[2]
        orig_mid   = blk_mid.forward
        def wrap_mid(self_blk, hidden_states, *a, _orig=orig_mid, **kw):
            h = _orig(hidden_states, *a, **kw)
            if self._style is not None:
                h = h + self._proj(2, h)
            return h
        blk_mid.forward = wrap_mid.__get__(blk_mid, blk_mid.__class__)
        # up blocks 0 & 1
        for off, (blk, _) in enumerate(self.sites[3:]):
            idx = 3 + off
            orig = blk.forward
            def wrap_up(self_blk, hidden_states, res_hidden_states_tuple, *a,
                        _orig=orig, _idx=idx, **kw):
                h = _orig(hidden_states, res_hidden_states_tuple, *a, **kw)
                if self._style is not None:
                    h = h + self._proj(_idx, h)
                return h
            blk.forward = wrap_up.__get__(blk, blk.__class__)

    @torch.no_grad()
    def generate(self, content, style, *, steps=20, guidance=1.0, keep=0.8, generator=None):
        content, style = content.to(self.device), style.to(self.device)
        fft_map    = self.cond(content.float(), style.float())
        self._style = (fft_map * guidance).to(self.dtype)
        post        = self.pipe.vae.encode((content * 2 - 1).to(self.dtype))
        dist        = post.latent_dist
        content_lat = dist.mean if hasattr(dist, "mean") else dist.distribution.loc
        content_lat = content_lat * self.pipe.vae.config.scaling_factor
        noise_lat = torch.randn(
            content_lat.shape,
            generator=generator,
            device=content_lat.device,
            dtype=content_lat.dtype
        )
        latents = content_lat * keep + noise_lat * (1.0 - keep)
        emb = self.pipe.text_encoder(
            self.pipe.tokenizer("", return_tensors="pt")
                .input_ids
                .to(self.device)
        )[0].to(self.dtype)
        out = self.pipe(
            prompt_embeds        = emb,
            latents              = latents,
            height               = content.shape[-2],
            width                = content.shape[-1],
            num_inference_steps  = steps,
            guidance_scale       = 1.0,
            generator            = generator,
        )
        self._style = None
        return out.images

# FFTStyleConditioner patch (auto-tiling fix, optional verbose)
import torch.nn as nn
DBG_FFT = False

def _dbg_stats(t):
    t32 = t.float()
    return (f"sh={tuple(t.shape)} "
            f"min={t32.min():+.3e} max={t32.max():+.3e} "
            f"mean={t32.mean():+.3e} std={t32.std():+.3e} "
            f"finite={torch.isfinite(t32).all().item()}")

def _fixed_forward(self, content, style, *, _eps=1e-6):
    c, s = content.float(), style.float()
    Cf, Sf = torch.fft.rfftn(c, dim=(-2, -1)), torch.fft.rfftn(s, dim=(-2, -1))
    Cm, Sm = torch.abs(Cf).clamp_min(_eps), torch.abs(Sf).clamp_min(_eps)
    R      = torch.log(Sm) - torch.log(Cm)
    mu     = R.mean(dim=(-2, -1), keepdim=True)
    sd     = R.std (dim=(-2, -1), keepdim=True, unbiased=False)
    Rn     = (R - mu) / (sd + _eps)
    x = Rn.mean(dim=1, keepdim=True)
    seq = next((m for m in self._modules.values() if isinstance(m, nn.Sequential)), None)
    first_conv = seq[0] if seq and isinstance(seq[0], nn.Conv2d) \
                     else next(m for m in self.modules() if isinstance(m, nn.Conv2d))
    need_C = first_conv.in_channels
    if x.shape[1] != need_C:
        reps = need_C // x.shape[1]
        if need_C % x.shape[1] != 0:
            raise RuntimeError(f"Cannot tile {x.shape[1]}→{need_C} channels cleanly")
        x = x.repeat(1, reps, 1, 1)
    if seq is None:
        for m in self._modules.values():
            x = m(x)
    else:
        for layer in seq:
            x = layer(x)
    return x

def _debug_forward(self, content, style, *, _eps=1e-6):
    print("\n▶︎  FFTStyleConditioner.forward()  ◀︎")
    step = 0
    def p(tag, tensor):
        nonlocal step
        step += 1
        print(f"  [{step:02d}] {tag:18s}", _dbg_stats(tensor))
    c, s = content.float(), style.float()
    p("content in", c); p("style in", s)
    Cf, Sf = torch.fft.rfftn(c, dim=(-2, -1)), torch.fft.rfftn(s, dim=(-2, -1))
    Cm, Sm = torch.abs(Cf).clamp_min(_eps), torch.abs(Sf).clamp_min(_eps)
    p("C mag", Cm); p("S mag", Sm)
    Cl, Sl = torch.log(Cm), torch.log(Sm)
    p("C log", Cl); p("S log", Sl)
    R  = Sl - Cl
    p("spectral res", R)
    mu = R.mean(dim=(-2, -1), keepdim=True)
    sd = R.std (dim=(-2, -1), keepdim=True, unbiased=False)
    Rn = (R - mu) / (sd + _eps)
    p("residual μ", mu); p("residual σ", sd); p("norm resid", Rn)
    x = Rn.mean(dim=1, keepdim=True)
    p("collapse RGB", x)
    seq = next((m for m in self._modules.values() if isinstance(m, nn.Sequential)), None)
    first_conv = seq[0] if seq and isinstance(seq[0], nn.Conv2d) \
                     else next(m for m in self.modules() if isinstance(m, nn.Conv2d))
    need_C = first_conv.in_channels
    if x.shape[1] != need_C:
        reps = need_C // x.shape[1]
        x = x.repeat(1, reps, 1, 1)
        p(f"tiled ×{reps}", x)
    if seq is None:
        for name, m in self._modules.items():
            x = m(x); p(name, x)
    else:
        for i, layer in enumerate(seq):
            x = layer(x); p(f"proj[{i}]", x)
    return x

if not hasattr(FFTStyleConditioner, "_orig_forward"):
    FFTStyleConditioner._orig_forward = FFTStyleConditioner.forward

def _router_forward(self, content, style, **kw):
    if DBG_FFT:
        return _debug_forward(self, content, style, **kw)
    return _fixed_forward(self, content, style, **kw)

FFTStyleConditioner.forward = _router_forward
