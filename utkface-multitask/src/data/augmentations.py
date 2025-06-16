import albumentations as A


def get_contrastive_augmentations(img_size: int):
    return A.Compose(
        [
            A.RandomResizedCrop((img_size, img_size), scale=(0.85, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            # imagenet normalization
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
        ]
    )
