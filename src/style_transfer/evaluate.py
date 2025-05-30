# evaluate.py
"""
Evaluation logic for FID and LPIPS metrics.
"""
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
from torchvision.transforms import Resize, CenterCrop, functional as TF
from PIL import Image

def evaluate_fid_lpips(model, dataset, n_samples=100, device=None):
    """
    Generate n_samples stylized images and compute FID and LPIPS against content set.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_metric   = FrechetInceptionDistance(feature=2048).to(device)
    lpips_metric = lpips.LPIPS(net='vgg').to(device)
    resize_299 = Resize(299, antialias=True)
    crop_299   = CenterCrop(299)
    model.eval() if hasattr(model, "eval") else None
    with torch.no_grad():
        for i in range(n_samples):
            ct, st = dataset[i]
            pil_img = model.generate(ct.unsqueeze(0).to(device),
                                     st.unsqueeze(0).to(device),
                                     steps=20)[0]
            pil_img = crop_299(resize_299(pil_img))
            img_t   = TF.pil_to_tensor(pil_img)
            fid_metric.update(img_t.unsqueeze(0).to(device), real=False)
            lpips_val = lpips_metric(
                TF.to_tensor(pil_img).unsqueeze(0).to(device),
                (ct.unsqueeze(0).to(device) * 0.5 + 0.5)
            ).item()
        content_root = [dataset.content[i] for i in range(n_samples)]
        for p in content_root:
            pil_img = Image.open(p).convert("RGB")
            pil_img = crop_299(resize_299(pil_img))
            img_t   = TF.pil_to_tensor(pil_img)
            fid_metric.update(img_t.unsqueeze(0).to(device), real=True)
    fid_val = fid_metric.compute().item()
    # LPIPS is averaged over fake images
    return fid_val
