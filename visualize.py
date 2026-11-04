"""Save an image grid to sanity-check the data, the augmentations and the predictions.

Check data/augmentations:  uv run python visualize.py --split train
Check predictions:         uv run python visualize.py --split val --run outputs/baseline
"""
import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from augmentation import MEAN, STD, get_train_transforms, get_val_transforms
from dataset import MarsDataset
from evaluate_test import load_model

# one RGB colour per class: soil, bedrock, sand, big_rock
COLORS = np.array(
    [[181, 134, 84], [70, 130, 180], [240, 210, 90], [220, 50, 50]], dtype=np.float32
)


def overlay(img, mask, alpha=0.55):
    """Tint pixels that have a valid class; 255 pixels keep the original image."""
    out = img.astype(np.float32).copy()
    valid = mask < len(COLORS)
    out[valid] = (1 - alpha) * out[valid] + alpha * COLORS[mask[valid]]
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--run", default=None, help="if given, also show that run's predictions")
    ap.add_argument("--n", type=int, default=6, help="number of images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d, size = cfg["data"], cfg["img_size"]
    split_file = Path(d["splits_dir"]) / f"{args.split}.txt"
    names = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
    names = random.Random(args.seed).sample(names, args.n)
    # the train split uses the random augmentations: we see exactly what the model sees
    tf = get_train_transforms(size) if args.split == "train" else get_val_transforms(size)
    ds = MarsDataset(d["train_img_dir"], d["train_mask_dir"], names, tf, d["mask_suffix"])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(args.run, device)[0] if args.run else None

    cols = 3 if model else 2
    fig, axes = plt.subplots(args.n, cols, figsize=(4 * cols, 4 * args.n), squeeze=False)
    for i in range(args.n):
        x, mask = ds[i]
        # undo the normalisation to get a displayable image back
        img = np.clip((x.permute(1, 2, 0).numpy() * STD + MEAN) * 255, 0, 255).astype(np.uint8)
        mask = mask.numpy()
        first = "image (after augmentation)" if args.split == "train" else "image"
        panels = [(first, img), ("ground truth", overlay(img, mask))]
        if model:
            with torch.no_grad():
                pred = model(x[None].to(device)).argmax(1)[0].cpu().numpy()
            panels.append(("prediction", overlay(img, pred)))
        for j, (title, panel) in enumerate(panels):
            axes[i, j].imshow(panel)
            axes[i, j].set_title(f"{title}\n{names[i]}" if j == 0 else title, fontsize=7)
            axes[i, j].axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c / 255) for c in COLORS]
    fig.legend(handles, cfg["class_names"], loc="lower center", ncol=4)
    out = Path(args.out or f"outputs/viz_{args.split}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
