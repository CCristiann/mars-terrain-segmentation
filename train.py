import argparse
import csv
import random
import shutil
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentation import get_train_transforms, get_val_transforms
from dataset import MarsDataset
from metrics import evaluate


def read_names(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def make_loader(ds, cfg, shuffle):
    nw = cfg["train"]["num_workers"]
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle,
                      num_workers=nw, persistent_workers=nw > 0, drop_last=shuffle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    run_dir = Path("outputs") / (args.run_name or time.strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True)
    shutil.copy(args.config, run_dir / "config.yaml")   # la run resta riproducibile

    d, t, size = cfg["data"], cfg["train"], cfg["img_size"]
    splits = Path(d["splits_dir"])
    common = (d["train_img_dir"], d["train_mask_dir"])
    train_ds = MarsDataset(*common, read_names(splits / "train.txt"),
                           get_train_transforms(size), d["mask_suffix"])
    val_ds = MarsDataset(*common, read_names(splits / "val.txt"),
                         get_val_transforms(size), d["mask_suffix"])
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)

    m = cfg["model"]
    model = getattr(smp, m["arch"])(encoder_name=m["encoder"],
                                    encoder_weights=m["encoder_weights"],
                                    in_channels=3, classes=cfg["n_classes"]).to(device)

    weights = torch.tensor(t["class_weights"], dtype=torch.float32, device=device)
    ce = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=cfg["ignore_index"])
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=cfg["ignore_index"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])

    with open(run_dir / "log.csv", "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_miou",
                                *[f"iou_{n}" for n in cfg["class_names"]]])

    best = -1.0
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        total = 0.0
        for images, masks in tqdm(train_loader, desc=f"epoch {epoch}"):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = ce(logits, masks) + t["dice_weight"] * dice(logits, masks)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()

        cm = evaluate(model, val_loader, device, cfg["n_classes"])
        miou, ious = cm.miou(), cm.iou().tolist()
        print(f"epoch {epoch}: loss {total / len(train_loader):.4f} "
              f"val mIoU {miou:.4f} | " +
              " ".join(f"{n} {v:.3f}" for n, v in zip(cfg["class_names"], ious)))
        with open(run_dir / "log.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, total / len(train_loader), miou, *ious])
        if miou > best:
            best = miou
            torch.save(model.state_dict(), run_dir / "best.pt")


if __name__ == "__main__":
    main()
