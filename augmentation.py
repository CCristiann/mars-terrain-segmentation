import albumentations as A
from albumentations.pytorch import ToTensorV2

# Mean and standard deviation of the ImageNet dataset. The encoder was pre-trained on
# ImageNet images normalised with these numbers, so we must feed it the same kind of input.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_train_transforms(size=512):
    """Training: random changes (augmentation) so the model sees a slightly different
    version of each image every epoch. This reduces overfitting (memorising the images)."""
    return A.Compose([
        # take a random part (60-100% of the area) and resize it to size x size
        A.RandomResizedCrop(size=(size, size), scale=(0.6, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),  # mirror left-right half of the time
        A.RandomBrightnessContrast(p=0.5),  # random light conditions
        A.Normalize(mean=MEAN, std=STD),  # pixel values 0-255 -> numbers around 0
        ToTensorV2(),  # numpy array (H, W, 3) -> PyTorch tensor (3, H, W)
    ])


def get_val_transforms(size=512):
    """Validation: no randomness, otherwise the measured quality would change every time."""
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
