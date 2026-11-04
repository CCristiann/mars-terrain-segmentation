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


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    cm = ConfusionMatrix(n_classes)
    for images, masks in loader:
        logits = model(images.to(device))
        cm.update(logits.argmax(1), masks)
    return cm
