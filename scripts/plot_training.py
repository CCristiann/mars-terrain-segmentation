"""Plot the training curves of a run (reads <run>/log.csv).

Usage:  uv run python scripts/plot_training.py --run outputs/quick --out docs/training_curve.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt

# same colours as visualize.py
COLORS = {"soil": "#b58654", "bedrock": "#4682b4", "sand": "#f0d25a", "big_rock": "#dc3232"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder, e.g. outputs/quick")
    ap.add_argument("--out", default="docs/training_curve.png")
    args = ap.parse_args()

    with open(Path(args.run) / "log.csv") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]

    fig, (ax_loss, ax_iou) = plt.subplots(1, 2, figsize=(10, 3.6))

    # left: the training loss should go down
    ax_loss.plot(epochs, [float(r["train_loss"]) for r in rows], marker="o", color="#444444")
    ax_loss.set_title("Training loss")
    ax_loss.set_xlabel("epoch")

    # right: validation IoU per class and their mean (mIoU), higher is better
    for name, color in COLORS.items():
        ax_iou.plot(epochs, [float(r[f"iou_{name}"]) for r in rows], marker="o", color=color, label=name)
    ax_iou.plot(epochs, [float(r["val_miou"]) for r in rows], "k--", marker="o", label="mIoU")
    ax_iou.set_title("Validation IoU")
    ax_iou.set_xlabel("epoch")
    ax_iou.set_ylim(0, 1)
    ax_iou.legend(loc="lower right", fontsize=8)

    for ax in (ax_loss, ax_iou):
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("saved:", out)


if __name__ == "__main__":
    main()
