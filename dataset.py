from pathlib import Path

import cv2
from torch.utils.data import Dataset


class MarsDataset(Dataset):
    """Gives the training loop one (image, mask) pair at a time.

    The mask is a picture with the same size as the image where every pixel holds
    the class number of that pixel: 0 soil, 1 bedrock, 2 sand, 3 big_rock, 255 = unlabelled.
    """

    def __init__(self, img_dir, mask_dir, names, transform, mask_suffix=""):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.names = names  # image file names, e.g. NLA_..._M1.JPG
        self.transform = transform  # augmentation + conversion to a tensor (see augmentation.py)
        self.mask_suffix = mask_suffix  # the mask of "X.JPG" is "X<suffix>.png"

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = self.img_dir / name
        mask_path = self.mask_dir / f"{Path(name).stem}{self.mask_suffix}.png"

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV reads BGR, models expect RGB

        # IMREAD_UNCHANGED: read the numbers exactly as stored (do not touch the 255)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(mask_path)
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise ValueError(f"mask does not match image: {mask_path} {mask.shape}")

        # the same random crop/flip is applied to image and mask, so they stay aligned
        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].long()
