"""Small helpers shared by several scripts (training, evaluation, visualization)."""
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import yaml


def get_device():
    """Pick the fastest hardware available: NVIDIA GPU (cuda), Apple GPU (mps) or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_names(path):
    """Read a split file (data/splits/train.txt or val.txt): one image file name per line."""
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def build_model(cfg, encoder_weights):
    """Create the segmentation network described in the config.

    encoder_weights="imagenet" starts the encoder from a network already trained on
    ImageNet (much faster than starting from random numbers); None means random weights.
    """
    m = cfg["model"]
    arch = getattr(smp, m["arch"])  # e.g. smp.Unet
    return arch(
        encoder_name=m["encoder"],
        encoder_weights=encoder_weights,
        in_channels=3,  # RGB (the grayscale NavCam images are repeated on 3 channels)
        classes=cfg["n_classes"],  # one output score per class, for every pixel
    )


def load_trained_model(run_dir, device):
    """Rebuild the model of a finished run and load its best weights (best.pt).

    Returns (model in eval mode, the config the run was trained with).
    """
    run_dir = Path(run_dir)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model = build_model(cfg, encoder_weights=None)  # the real weights come from best.pt
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
    return model.to(device).eval(), cfg
