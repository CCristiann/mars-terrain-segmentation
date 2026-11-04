"""Make the encoder visible: what does the network "see" at each level?

Creates two pictures in outputs/explain/:
  encoder_levels.png   the image shrinking from 512x512 to 16x16, level by level
  cell_similarity.png  what one 16x16 cell "knows": where else does the model see the same thing?

Usage:  uv run python explain_encoder.py --run outputs/quick
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

from augmentation import MEAN, STD, get_val_transforms
from dataset import MarsDataset
from utils import get_device, load_trained_model, read_names
from visualize import overlay


def features_to_rgb(feat):
    """Squash a feature map (channels x H x W) into a 3-colour picture.

    Each cell holds a list of C numbers. PCA finds the 3 directions in which these lists
    differ the most and uses them as R, G, B. The colours have NO fixed meaning: they only say
    "cells with similar colour hold similar information" (inside this image).
    """
    c, h, w = feat.shape
    x = feat.reshape(c, -1).T  # one row per cell, one column per channel
    x = x - x.mean(0, keepdim=True)
    _, _, vt = torch.linalg.svd(x, full_matrices=False)
    comp = (x @ vt[:3].T).numpy()  # (cells, 3)
    lo, hi = np.percentile(comp, 2, axis=0), np.percentile(comp, 98, axis=0)
    comp = np.clip((comp - lo) / (hi - lo + 1e-8), 0, 1)
    return comp.reshape(h, w, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/quick")
    ap.add_argument("--name", default="NLB_460433266EDR_F0400320NCAM00479M1.JPG",
                    help="validation image to explain (a sand dune with a patch of soil)")
    ap.add_argument("--cells", default="12,5;2,12;8,13", help="row,col of the 16x16 cells to inspect")
    ap.add_argument("--out-dir", default="outputs/explain")
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_trained_model(args.run, device)
    d, size = cfg["data"], cfg["img_size"]

    name = args.name
    ds = MarsDataset(d["train_img_dir"], d["train_mask_dir"], [name], get_val_transforms(size), d["mask_suffix"])
    x, mask = ds[0]
    img = np.clip((x.permute(1, 2, 0).numpy() * STD + MEAN) * 255, 0, 255).astype(np.uint8)

    with torch.no_grad():
        feats = [f[0].cpu() for f in model.encoder(x[None].to(device))]  # level 0 = the input itself
        pred = model(x[None].to(device)).argmax(1)[0].cpu().numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- picture 1: the encoder level by level ----------
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    axes[0].imshow(img)
    axes[0].set_title(f"INPUT\n{size}x{size} cells, 3 numbers each (R,G,B)", fontsize=10)
    for level in range(1, 6):
        f = feats[level]
        c, h, w = f.shape
        axes[level].imshow(features_to_rgb(f), interpolation="nearest")  # nearest: show the real cells
        axes[level].set_title(f"LEVEL {level}\n{h}x{w} cells, {c} numbers each\n1 cell = {size // h}x{size // h} input pixels", fontsize=10)
    axes[6].imshow(overlay(img, mask.numpy()))
    axes[6].set_title("human labels (for comparison)", fontsize=10)
    axes[7].imshow(overlay(img, pred))
    axes[7].set_title("final prediction of the model", fontsize=10)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"What the encoder does to {name}\n(colours = PCA of the numbers in each cell: similar colour = similar content)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=3)
    fig.savefig(out_dir / "encoder_levels.png", dpi=90)
    plt.close(fig)

    # ---------- picture 2: what does one 16x16 cell know? ----------
    f5 = feats[5]  # (512, 16, 16)
    c, h, w = f5.shape
    cell_px = size // h
    vec = F.normalize(f5.reshape(c, -1).T, dim=1)  # one unit-length vector (512 numbers) per cell
    cells = [tuple(int(v) for v in s.split(",")) for s in args.cells.split(";")]
    colors = ["#ff00ff", "#00e5ff", "#ffffff"]

    fig, axes = plt.subplots(2, len(cells), figsize=(5 * len(cells), 11), squeeze=False)
    fig.subplots_adjust(hspace=0.25, wspace=0.08)
    for j, ((row, col), color) in enumerate(zip(cells, colors)):
        # cosine similarity between this cell's 512 numbers and those of every other cell:
        # 1 = same kind of content, 0 = unrelated
        sim = (vec @ vec[row * w + col]).reshape(h, w).numpy()

        ax = axes[0, j]
        ax.imshow(img)
        for k in range(1, h):  # the 16x16 grid
            ax.axhline(k * cell_px - 0.5, color="white", lw=0.3, alpha=0.5)
            ax.axvline(k * cell_px - 0.5, color="white", lw=0.3, alpha=0.5)
        ax.add_patch(Rectangle((col * cell_px - 0.5, row * cell_px - 0.5), cell_px, cell_px,
                               fill=False, ec=color, lw=3))
        ax.set_title(f"chosen cell (row {row}, col {col})\n= a {cell_px}x{cell_px} px square of the input", fontsize=10)
        ax.axis("off")

        ax = axes[1, j]
        ax.imshow(img, alpha=0.35)
        im = ax.imshow(sim, cmap="inferno", vmin=0, vmax=1, alpha=0.8, interpolation="nearest",
                       extent=(-0.5, size - 0.5, size - 0.5, -0.5))
        ax.add_patch(Rectangle((col * cell_px - 0.5, row * cell_px - 0.5), cell_px, cell_px,
                               fill=False, ec=color, lw=3))
        ax.set_title("similarity of every cell\nwith the chosen cell", fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes[1, :], shrink=0.6, label="similarity with the chosen cell")
    fig.suptitle(f"One cell of level 5 = {c} numbers. Cells with similar numbers = similar terrain", fontsize=12)
    fig.savefig(out_dir / "cell_similarity.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out_dir / "encoder_levels.png", "and", out_dir / "cell_similarity.png")


if __name__ == "__main__":
    main()
