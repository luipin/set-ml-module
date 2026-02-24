import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_spatial_color_transforms():
    """
    Returns spatial and color augmentations without normalization or tensor conversion.

    These transforms are designed to simulate real-world variance while preserving
    the core features of the Set cards.

    Augmentation Choices:
    - ShiftScaleRotate: Simulates cards being slightly off-center, tilted, or at different distances.
      `shift_limit=0.05` keeps the card mostly within frame. `rotate_limit=45` covers most casual
      orientations without requiring a full 360-degree search.
    - RandomBrightnessContrast: Simulates different indoor lighting conditions.
    - HueSaturationValue: CRITICAL. We use strict limits (+/- 15) for hue to ensure a 'red'
      card doesn't accidentally become 'purple' or 'green' due to augmentation.
    - GaussianBlur: Simulates slight camera out-of-focus or motion blur.

    Returns:
        albumentations.Compose: A composition of spatial and color transforms.
    """
    return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        # STRICT LIMITS: +/- 15 for Hue to maintain color label integrity (Red, Green, Purple)
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=0.7),
        A.GaussianBlur(blur_limit=3, p=0.3),
    ])

def get_train_transforms():
    """
    Returns the full Albumentations pipeline for training data, including ImageNet Normalization.

    The normalization constants [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] are standard
    for models pre-trained on ImageNet (like our ResNet18 backbone).

    Returns:
        albumentations.Compose: The full training transformation pipeline.
    """
    return A.Compose([
        get_spatial_color_transforms(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    """
    Returns Albumentations pipeline for validation/test data.

    Only includes normalization and tensor conversion. No spatial or color augmentations
    are applied to ensure we evaluate the model on 'clean' (albeit normalized) data.

    Returns:
        albumentations.Compose: The validation transformation pipeline.
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
