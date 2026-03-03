import cv2
import numpy as np
import random
import os
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AddRandomBackground(A.ImageOnlyTransform):
    """
    Custom transform that replaces black/dark padding (introduced by rotation/scaling) 
    and the dark corners of the raw seed images with random colors, noise, or real textures.
    """
    def __init__(self, bg_dir=None, p=0.5):
        super().__init__(p=p)
        self.bg_dir = bg_dir
        self.bg_images = []
        if self.bg_dir and os.path.exists(self.bg_dir):
            self.bg_images = glob.glob(os.path.join(self.bg_dir, "*.[pj][pn][g]*"))
            
    def apply(self, img, **params):
        # 1. Identify background pixels.
        # Raw seed images have dark corners (< 30). ShiftScaleRotate introduces black padding (0).
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        
        # 2. Generate a random background
        h, w, c = img.shape
        bg = np.zeros_like(img)
        
        if self.bg_images and random.random() < 0.7:
            # 70% chance to use a real background image if a directory is provided
            bg_path = random.choice(self.bg_images)
            bg_img = cv2.imread(bg_path)
            if bg_img is not None:
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
                bg = cv2.resize(bg_img, (w, h))
        else:
            # Generate a random solid color or static noise
            if random.random() < 0.5:
                color = np.random.randint(50, 255, 3, dtype=np.uint8)
                bg[:] = color
            else:
                bg = np.random.randint(50, 255, (h, w, c), dtype=np.uint8)
                
        # 3. Blend the background into the masked areas
        mask_3d = mask[:, :, None].astype(bool)
        result = np.where(mask_3d, bg, img)
        return result

def get_spatial_color_transforms(bg_dir=None):
    """
    Returns spatial and color augmentations without normalization or tensor conversion.

    These transforms are designed to simulate real-world variance while preserving
    the core features of the Set cards.

    Augmentation Choices:
    - AddRandomBackground: Replaces empty padded space with textures or colors.
    - ShiftScaleRotate: Simulates cards being slightly off-center, tilted, or at different distances.
      `border_mode=cv2.BORDER_CONSTANT` ensures padded areas are solid black for the background replacement.
    - RandomBrightnessContrast: Simulates different indoor lighting conditions.
    - HueSaturationValue: CRITICAL. We use strict limits (+/- 15) for hue to ensure a 'red'
      card doesn't accidentally become 'purple' or 'green' due to augmentation.
    - GaussianBlur: Simulates slight camera out-of-focus or motion blur.

    Returns:
        albumentations.Compose: A composition of spatial and color transforms.
    """
    return A.Compose([
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=45, 
            border_mode=cv2.BORDER_CONSTANT, value=0, p=0.7
        ),
        AddRandomBackground(bg_dir=bg_dir, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        # STRICT LIMITS: +/- 15 for Hue to maintain color label integrity (Red, Green, Purple)
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=30, p=0.7),
        A.GaussianBlur(blur_limit=3, p=0.3),
    ])

def get_train_transforms():
    """
    Returns the full Albumentations pipeline for training data, including ImageNet Normalization.

    Resize comes first to ensure all images are 224x224 before augmentation. ResNet18
    requires this exact input size. Resizing before spatial transforms (not after) means
    ShiftScaleRotate and friends always operate at the target resolution.

    The normalization constants [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] are standard
    for models pre-trained on ImageNet (like our ResNet18 backbone).

    Returns:
        albumentations.Compose: The full training transformation pipeline.
    """
    return A.Compose([
        A.Resize(224, 224),
        get_spatial_color_transforms(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    """
    Returns Albumentations pipeline for validation/test data.

    Only includes resize, normalization, and tensor conversion. No spatial or color
    augmentations are applied to ensure we evaluate the model on 'clean' (albeit
    normalized) data.

    Returns:
        albumentations.Compose: The validation transformation pipeline.
    """
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
