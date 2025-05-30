# dataset.py
"""
Dataset and transforms for style transfer pairs.
"""
import pathlib
import random
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode

def drop_alpha(img_tensor):
    """Drop alpha channel if present."""
    return img_tensor[:3]

transform = Compose([
    drop_alpha,
    Resize(384, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(384),
    lambda x: x.float().div(255.),
    Normalize(0.5, 0.5),
])

class StyleTransferPairs(Dataset):
    """
    Returns (content, style) float tensors in [-1,1] and exactly 3 channels.
    Folder layout:
        data_root/
            content/*.jpg|png
            style/<artist>/*.jpg|png
    """
    def __init__(self, data_root, size=384):
        root = pathlib.Path(data_root)
        self.content = sorted((root/"content").rglob("*.[jp][pn]g"))
        self.style   = sorted((root/"style").rglob("*.[jp][pn]g"))
        self.tfm = Compose([
            drop_alpha,
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            lambda x: x.float().div(255.),
            Normalize(0.5, 0.5),
        ])
    def __len__(self):
        return len(self.content)
    def __getitem__(self, idx):
        ct = self.tfm(torchvision.io.read_image(str(self.content[idx])))
        st = self.tfm(torchvision.io.read_image(str(random.choice(self.style))))
        return ct, st
