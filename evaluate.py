import argparse
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import yaml
from torch.utils.data import DataLoader

from augmentation import get_val_transforms
from dataset import MarsDataset
from metrics import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--mask-dir", type=Path, required=True)
    ap.add_argument("--mask-suffix", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.run_dir / "config.yaml"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    m = cfg["model"]
    model = getattr(smp, m["arch"])(
        encoder_name=m["encoder"],
        encoder_weights=None,
        in_channels=3,
        classes=cfg["n_classes"],
    )

    model.load_state_dict(
        torch.load(args.run_dir / "best.pt", map_location=device, weights_only=True)
    )
    model.to(device)

    names = sorted(
        p.name
        for p in args.img_dir.iterdir()
        if (args.mask_dir / f"{p.stem} {args.mask_suffix}.png").exists()
    )

    ds = MarsDataset(args.img_dir, args.mask_dir, names, get_val_transforms(cfg["img_size"]), args.mask_suffix)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    cm = evaluate(model, loader, device, cfg["n_classes"])
    for n, v in zip(cfg["class_names"], cm.iou().tolist()):
        print(f"IoU {n:10s} {v:.4f}")
    
    print(f"mIoU {cm.miou():.4f}")


if __name__ == "__main__":
    main()
