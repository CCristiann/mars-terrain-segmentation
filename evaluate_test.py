"""Final evaluation of a run on the expert-labelled test set (322 images), at native resolution.

Usage:  uv run python evaluate_test.py --run outputs/baseline
"""
import argparse
from pathlib import Path

import albumentations as A
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from augmentation import MEAN, STD
from dataset import MarsDataset
from metrics import ConfusionMatrix

# How many experts must agree on a pixel for it to be kept (min3 = all three, the strictest).
VARIANTS = ["min1", "min2", "min3"]


def load_model(run_dir, device):
    """Rebuild the model described by the run's config and load its best weights."""
    cfg = yaml.safe_load(open(Path(run_dir) / "config.yaml"))
    m = cfg["model"]
    model = getattr(smp, m["arch"])(
        encoder_name=m["encoder"],
        encoder_weights=None,  # the real weights come from best.pt, no ImageNet download
        in_channels=3,
        classes=cfg["n_classes"],
    )
    model.load_state_dict(
        torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=True)
    )
    return model.to(device).eval(), cfg


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder, e.g. outputs/baseline")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, cfg = load_model(args.run, device)
    size, n = cfg["img_size"], cfg["n_classes"]

    img_dir = cfg["data"]["train_img_dir"]
    test_root = Path(cfg["data"]["train_mask_dir"]).parent / "test"
    # no resizing in the transform: we resize by hand below to keep the native mask
    tf = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

    out = []
    for v in args.variants:
        mask_dir = test_root / f"masked-gold-{v}-100agree"
        names = sorted(
            p.name.removesuffix("_merged.png") + ".JPG"
            for p in mask_dir.glob("*_merged.png")
        )
        ds = MarsDataset(img_dir, mask_dir, names, tf, mask_suffix="_merged")
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)

        cm = ConfusionMatrix(n)
        for images, masks in loader:
            # the model sees the same resolution as in training (e.g. 512)...
            x = F.interpolate(
                images.to(device), size=(size, size), mode="bilinear", align_corners=False
            )
            logits = model(x)
            # ...but predictions are compared with the original full-size mask
            logits = F.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            cm.update(logits.argmax(1), masks)
        out.append(f"=== test {v} ({len(names)} images) ===\n{cm.report(cfg['class_names'])}\n")

    text = "\n".join(out)
    print(text)
    (Path(args.run) / "test_report.txt").write_text(text)


if __name__ == "__main__":
    main()
