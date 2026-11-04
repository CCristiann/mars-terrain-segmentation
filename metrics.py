"""How we measure the quality of the segmentation: confusion matrix, IoU and mIoU."""
import torch


class ConfusionMatrix:
    """Counts, over many pixels, how often a true class was predicted as each class.

    mat[i][j] = number of pixels whose TRUE class is i and PREDICTED class is j.
    The diagonal holds the correct predictions.
    """

    def __init__(self, n_classes):
        self.n = n_classes
        self.mat = torch.zeros(n_classes, n_classes, dtype=torch.int64)

    def update(self, preds, target):
        preds, target = preds.flatten().cpu(), target.flatten().cpu()
        valid = target < self.n  # skip the pixels labelled 255 (no label)
        # trick: every (true, predicted) pair becomes one number = a cell of the matrix,
        # then bincount counts how many pixels fall in each cell
        idx = target[valid] * self.n + preds[valid]
        self.mat += torch.bincount(idx, minlength=self.n**2).reshape(self.n, self.n)

    def iou(self):
        """Intersection over Union for each class: TP / (TP + FP + FN).

        Of all the pixels that are that class in the truth OR in the prediction,
        what fraction is correct in both? 1.0 = perfect, 0.0 = completely wrong.
        """
        tp = self.mat.diag().float()  # predicted class c and truly class c
        # row sum = pixels truly of the class, column sum = pixels predicted as the class
        union = self.mat.sum(0).float() + self.mat.sum(1).float() - tp
        return torch.where(union > 0, tp / union, torch.full_like(tp, float("nan")))

    def miou(self):
        """Mean IoU: the average of the per-class IoUs (every class counts the same)."""
        return torch.nanmean(self.iou()).item()

    def report(self, class_names):
        """Readable text: per-class IoU and the confusion matrix (rows = true class, in %)."""
        lines = [
            "IoU: " + "  ".join(f"{n} {v:.3f}" for n, v in zip(class_names, self.iou().tolist())),
            f"mIoU: {self.miou():.4f}",
            "",
            "Confusion matrix (row = true class, column = predicted, % of the row):",
            " " * 10 + "".join(f"{n:>10s}" for n in class_names),
        ]
        rows = self.mat.float() / self.mat.sum(1, keepdim=True).clamp(min=1)
        for name, row in zip(class_names, rows.tolist()):
            lines.append(f"{name:>10s}" + "".join(f"{100 * v:10.1f}" for v in row))
        return "\n".join(lines)


@torch.no_grad()  # no gradients needed: we only measure, we do not learn
def evaluate(model, loader, device, n_classes, max_batches=None):
    """Run the model on every image of the loader and accumulate one confusion matrix.

    We accumulate over the whole dataset and compute IoU once at the end: averaging
    the IoU of single batches would give a distorted result.
    """
    model.eval()  # evaluation mode (e.g. BatchNorm uses its accumulated statistics)
    cm = ConfusionMatrix(n_classes)
    for i, (images, masks) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        logits = model(images.to(device))  # (batch, classes, H, W) scores
        cm.update(logits.argmax(1), masks)  # predicted class = the class with the highest score
    return cm
