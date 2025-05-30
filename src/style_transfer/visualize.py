# visualize.py
"""
Visualization utilities for training progress and results.
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

def plot_loss_curves(epochs, train_loss, val_loss):
    plt.figure(figsize=(5,4))
    plt.plot(epochs, train_loss, marker='o', label='Train')
    plt.plot(epochs, val_loss,   marker='o', label='Val')
    plt.title("Adapter Fine-Tune Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.xticks(epochs); plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.show()

def plot_fid_lpips(epochs, fid_vals, lpips_vals):
    fig, ax1 = plt.subplots(figsize=(5,4))
    ax1.plot(epochs, fid_vals,   marker='s', label='FID')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("FID")
    ax2 = ax1.twinx()
    ax2.plot(epochs, lpips_vals, marker='^', linestyle='--', label='LPIPS')
    ax2.set_ylabel("LPIPS")
    ax1.set_xticks(epochs)
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labs1+labs2, loc='upper right')
    plt.title("Perceptual Metrics Over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_gallery(rows, nrow=3, filename="gallery.png"):
    grid = make_grid(rows, nrow=nrow)
    arr  = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(filename)
    print(f"🖼️  Saved → {filename}")
