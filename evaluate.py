"""Evaluate a trained run on ANY folder of images + masks (generic and simple).

Images and masks are both resized to the training size (e.g. 512), exactly like the validation.
For the official test-set numbers (full resolution, 3 expert variants) use evaluate_test.py.

Example:
    uv run python evaluate.py --run-dir outputs/quick \
        --img-dir data/raw/ai4mars-dataset-merged-0.6/msl/ncam/images/edr \
        --mask-dir data/raw/ai4mars-dataset-merged-0.6/msl/ncam/labels/test/masked-gold-min3-100agree \
        --mask-suffix _merged
"""
import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from augmentation import get_val_transforms
from dataset import MarsDataset
from metrics import evaluate
from utils import get_device, load_trained_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--mask-dir", type=Path, required=True)
    ap.add_argument("--mask-suffix", default="")
    args = ap.parse_args()

    device = get_device()
    model, cfg = load_trained_model(args.run_dir, device)

    # keep only the images that have a mask
    names = sorted(
        p.name
        for p in args.img_dir.iterdir()
        if (args.mask_dir / f"{p.stem}{args.mask_suffix}.png").exists()
    )

    ds = MarsDataset(args.img_dir, args.mask_dir, names, get_val_transforms(cfg["img_size"]), args.mask_suffix)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    cm = evaluate(model, loader, device, cfg["n_classes"])
    for n, v in zip(cfg["class_names"], cm.iou().tolist()):
        print(f"IoU {n:10s} {v:.4f}")
    print(f"mIoU {cm.miou():.4f}")


if __name__ == "__main__":
    main()
