"""Loss functions. The loss is <pixel_term> + weight * <overlap_term>,
chosen in config.yaml with train.loss: "<pixel>_<overlap>" (e.g. ce_dice, ce_lovasz).
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

PIXEL_TERMS = ("ce", "focal")
OVERLAP_TERMS = ("dice", "tversky", "lovasz", "none")


class FocalLoss(torch.nn.Module):
    """Softmax focal loss: CE * (1-p)^gamma, p = probability given to the true class.

    Easy pixel (p near 1) -> factor near 0 -> barely counts. Hard pixel -> full loss.
    gamma = 0 gives back the plain weighted cross-entropy
    """

    def __init__(self, weight, gamma, ignore_index):
        super().__init__()
        self.register_buffer(
            "weight", weight
        )  # moves to the GPU together with the module
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        valid = target != self.ignore_index

        # per-pixel loss (not averaged yet), with class weights
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # probability of the true class, WITHOUT class weights: p = exp(-plain CE)
        p = torch.exp(
            -F.cross_entropy(
                logits, target, ignore_index=self.ignore_index, reduction="none"
            )
        )

        loss = (1 - p) ** self.gamma * ce

        w = self.weight[target.clamp(max=len(self.weight) - 1)] * valid

        return loss[valid].sum() / w.sum().clamp(min=1e-8)


def parse_loss_name(name):
    pixel, _, overlap = name.partition("_")
    if pixel not in PIXEL_TERMS or overlap not in OVERLAP_TERMS:
        raise ValueError(
            f"Unknown train.loss '{name}': use <pixel>_<overlap>, "
            f"pixel in {PIXEL_TERMS}, overlap in {OVERLAP_TERMS}"
        )

    return pixel, overlap


def make_loss(cfg, device):
    t, ii = cfg["train"], cfg["ignore_index"]
    pixel, overlap = parse_loss_name(t.get("loss", "ce_dice"))
    weights = torch.tensor(t["class_weights"], dtype=torch.float32, device=device)

    if pixel == "ce":
        pixel_fn = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=ii)
    else:
        pixel_fn = FocalLoss(weights, gamma=t["focal_gamma"], ignore_index=ii).to(
            device
        )

    if overlap == "dice":
        overlap_fn = smp.losses.DiceLoss(mode="multiclass", ignore_index=ii)
        overlap_weight = t["dice_weight"]
    elif overlap == "tversky":
        overlap_fn = smp.losses.TverskyLoss(
            mode="multiclass",
            ignore_index=ii,
            alpha=t["tversky_alpha"],
            beta=t["tversky_beta"],
        )
        overlap_weight = t["tversky_weight"]
    elif overlap == "lovasz":
        overlap_fn = smp.losses.LovaszLoss(mode="multiclass", ignore_index=ii)
        overlap_weight = t["lovasz_weight"]
    else:
        overlap_fn, overlap_weight = None, 0.0

    def loss_fn(logits, masks):
        logits = logits.float()  # loss always in 32 bit, even with mixed precision

        loss = pixel_fn(logits, masks)
        if overlap_fn is not None:
            loss = loss + overlap_weight * overlap_fn(logits, masks)

        return loss

    return loss_fn
