import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def get_train_transforms(size=512):
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.6, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

def get_val_transforms(size=512):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])