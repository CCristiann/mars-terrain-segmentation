"""Train a U-Net that labels every pixel of a Mars image: soil, bedrock, sand or big_rock.

Usage:
    uv run python train.py --run-name quick            # normal run, settings from config.yaml
    uv run python train.py --run-name check --smoke    # only 20 batches per epoch: tests the pipeline
    uv run python train.py --resume outputs/quick      # continue an interrupted run
"""
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
from utils import build_model, get_device, read_names

SMOKE_BATCHES = 20  # with --smoke: batches per epoch, for both training and validation


def make_loader(dataset, cfg, shuffle):
    """A DataLoader hands the model one batch (a small group of images) at a time.

    shuffle=True for training: seeing the images in a different order every epoch
    helps the model learn. drop_last=True skips the final incomplete batch.
    """
    workers = cfg["train"]["num_workers"]  # parallel processes that read and augment images
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def make_loss(cfg, device):
    """Loss = how wrong the model is. Training tries to make this number smaller.

    - CrossEntropy: per-pixel classification error. The class weights make mistakes on
      rare classes (big_rock) count more. Pixels labelled 255 are ignored.
    - Dice: measures the overlap between predicted and true regions per class; it helps
      small classes that CrossEntropy alone tends to neglect.
    """
    t = cfg["train"]
    weights = torch.tensor(t["class_weights"], dtype=torch.float32, device=device)
    ce = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=cfg["ignore_index"])
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=cfg["ignore_index"])

    def loss_fn(logits, masks):
        return ce(logits, masks) + t["dice_weight"] * dice(logits, masks)

    return loss_fn


def train_one_epoch(model, loader, loss_fn, opt, device, desc, max_batches=None):
    """One pass over the training images. Returns the average loss."""
    model.train()  # training mode: e.g. BatchNorm layers use the statistics of the batch
    total, steps = 0.0, 0
    for images, masks in tqdm(loader, desc=desc, total=max_batches or len(loader)):
        if max_batches is not None and steps >= max_batches:
            break
        images, masks = images.to(device), masks.to(device)

        logits = model(images)  # 1. forward: the model predicts a score per class per pixel
        loss = loss_fn(logits, masks)  # 2. compare predictions with the true masks
        opt.zero_grad(set_to_none=True)  # 3. forget the gradients of the previous batch
        loss.backward()  # 4. backward: how should each weight change to reduce the loss?
        opt.step()  # 5. update the weights a small step in that direction

        total += loss.item()
        steps += 1
    return total / steps


def save_checkpoint(run_dir, model, opt, sched, epoch, best):
    """Save everything needed to resume. Write to a temporary file and then rename it,
    so an interruption while saving cannot corrupt the last good checkpoint."""
    tmp = run_dir / "last.pt.tmp"
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
         "epoch": epoch, "best": best},
        tmp,
    )
    tmp.replace(run_dir / "last.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", default=None, help="folder of an interrupted run to resume")
    ap.add_argument("--smoke", action="store_true", help="quick check: only a few batches per epoch")
    args = ap.parse_args()

    # when resuming, use the config saved inside the run so that the scheduler
    # and all hyperparameters stay exactly the same
    cfg = yaml.safe_load(open(Path(args.resume) / "config.yaml" if args.resume else args.config))
    d, t, size = cfg["data"], cfg["train"], cfg["img_size"]

    # fixed random seeds: the run is (almost) repeatable
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = get_device()
    print("device:", device)

    # every run gets its own folder with a copy of the config used
    if args.resume:
        run_dir = Path(args.resume)
    else:
        run_dir = Path("outputs") / (args.run_name or time.strftime("%Y%m%d-%H%M%S"))
        run_dir.mkdir(parents=True)
        shutil.copy(args.config, run_dir / "config.yaml")

    # --- data: images + masks, split into training and validation ---
    # The model learns from the training images. The validation images are never used
    # for learning: they tell us how well the model works on images it has not seen.
    splits = Path(d["splits_dir"])
    paths = (d["train_img_dir"], d["train_mask_dir"])
    train_ds = MarsDataset(*paths, read_names(splits / "train.txt"), get_train_transforms(size), d["mask_suffix"])
    val_ds = MarsDataset(*paths, read_names(splits / "val.txt"), get_val_transforms(size), d["mask_suffix"])
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    val_loader = make_loader(val_ds, cfg, shuffle=False)

    # --- model, loss, optimizer, learning-rate schedule ---
    model = build_model(cfg, cfg["model"]["encoder_weights"]).to(device)
    loss_fn = make_loss(cfg, device)
    # AdamW decides how to update the weights; lr (learning rate) is the size of each step
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    # the learning rate shrinks smoothly (cosine curve) from lr to ~0 over all the epochs:
    # big steps at the start, fine adjustments at the end
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])

    start_epoch, best = 1, -1.0
    if args.resume:
        ck = torch.load(run_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resuming from epoch {start_epoch} (best mIoU so far {best:.4f})")
    else:
        with open(run_dir / "log.csv", "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_miou", *[f"iou_{n}" for n in cfg["class_names"]]])

    max_batches = SMOKE_BATCHES if args.smoke else None

    # --- main loop: one epoch = the model sees every training image once ---
    for epoch in range(start_epoch, t["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, opt, device,
                                     desc=f"epoch {epoch}", max_batches=max_batches)
        sched.step()  # lower the learning rate according to the cosine curve

        # after each epoch: measure quality on the validation images (no learning here)
        cm = evaluate(model, val_loader, device, cfg["n_classes"], max_batches=max_batches)
        miou, ious = cm.miou(), cm.iou().tolist()
        print(f"epoch {epoch}: loss {train_loss:.4f} val mIoU {miou:.4f} | "
              + " ".join(f"{n} {v:.3f}" for n, v in zip(cfg["class_names"], ious)))
        with open(run_dir / "log.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, miou, *ious])

        # keep the weights of the best epoch (by validation mIoU) in best.pt
        if miou > best:
            best = miou
            torch.save(model.state_dict(), run_dir / "best.pt")
        save_checkpoint(run_dir, model, opt, sched, epoch, best)  # last.pt, used by --resume


if __name__ == "__main__":
    main()
