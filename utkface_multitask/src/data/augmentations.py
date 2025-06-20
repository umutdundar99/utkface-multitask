import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_contrastive_augmentations(img_size: int):
    return A.Compose(
        [
            A.RandomResizedCrop((img_size, img_size), scale=(0.85, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.Resize(height=img_size, width=img_size, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
            ToTensorV2(),
        ]
    )


def get_classification_augmentations(img_size: int):
    return A.Compose(
        [
            A.RandomResizedCrop((img_size, img_size), scale=(0.85, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.CLAHE(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
            ToTensorV2(),
        ]
    )
