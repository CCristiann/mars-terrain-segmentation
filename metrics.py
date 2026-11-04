import torch


class ConfusionMatrix:
    def __init__(self, n_classes):
        self.n = n_classes
        self.mat = torch.zeros(n_classes, n_classes, dtype=torch.int64)

    def update(self, preds, target):
        preds, target = preds.flatten().cpu(), target.flatten().cpu()
        valid = target < self.n
        idx = target[valid] * self.n + preds[valid]
        self.mat += torch.bincount(idx, minlength=self.n**2).reshape(self.n, self.n)

    def iou(self):
        tp = self.mat.diag().float()
        union = self.mat.sum(0).float() + self.mat.sum(1).float() - tp
        return torch.where(union > 0, tp / union, torch.full_like(tp, float("nan")))

    def miou(self):
        return torch.nanmean(self.iou()).item()

    def report(self, class_names):
        """Readable text: per-class IoU and the confusion matrix (rows = true class, in %)."""
        lines = [
            "IoU: "
            + "  ".join(
                f"{n} {v:.3f}" for n, v in zip(class_names, self.iou().tolist())
            ),
            f"mIoU: {self.miou():.4f}",
            "",
            "Confusion matrix (row = true class, column = predicted, % of the row):",
            " " * 10 + "".join(f"{n:>10s}" for n in class_names),
        ]
        rows = self.mat.float() / self.mat.sum(1, keepdim=True).clamp(min=1)
        for name, row in zip(class_names, rows.tolist()):
            lines.append(f"{name:>10s}" + "".join(f"{100 * v:10.1f}" for v in row))
        return "\n".join(lines)


@torch.no_grad()
def evaluate(model, loader, device, n_classes, max_batches=None):
    model.eval()
    cm = ConfusionMatrix(n_classes)
    for i, (images, masks) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        logits = model(images.to(device))
        cm.update(logits.argmax(1), masks)
    return cm
