import os
import cv2
import glob
import argparse
from tqdm import tqdm
from src.utils.augmentations import get_spatial_color_transforms

def bootstrap_dataset(raw_dir: str, augmented_dir: str, augmentations_per_image: int = 300):
    """
    Reads the 81 seed images from raw_dir, applies offline Albumentation augmentations, 
    and saves them to augmented_dir (creating ~24,000 to ~40,000 images).

    The script enforces a strict check for exactly 81 unique card images to ensure 
    the full deck is represented before starting the augmentation process.

    Args:
        raw_dir (str): Directory with the 81 clean seed images.
        augmented_dir (str): Destination directory for the final flattened training set.
        augmentations_per_image (int): Number of variations to create per seed image.

    Raises:
        FileNotFoundError: If the raw_dir doesn't contain exactly 81 seed images.
    """
    os.makedirs(augmented_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(raw_dir, '*.jpg')) + glob.glob(os.path.join(raw_dir, '*.png'))
    
    # Milestone 0 requirement: Verify 81 seed images exist
    if len(image_paths) != 81:
        raise FileNotFoundError(
            f"Expected 81 seed images in {raw_dir}, but found {len(image_paths)}. "
            "The Set deck must be complete before bootstrapping."
        )

    # Use only spatial and color transforms (No ToTensorV2 or Normalization)
    transform = get_spatial_color_transforms()

    print(f"Verified 81 seed images. Generating {augmentations_per_image} variants each...")
    
    for img_path in tqdm(image_paths):
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        
        # Read the original image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read {img_path}. Skipping.")
            continue
        
        # Albumentations expects RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save the original unaugmented image
        cv2.imwrite(os.path.join(augmented_dir, f"{name}_orig{ext}"), image)

        # Generate N augmented variations
        for i in range(augmentations_per_image):
            augmented = transform(image=image_rgb)
            augmented_image = augmented['image']
            
            # Convert back to BGR for saving with OpenCV
            augmented_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
            
            # Use the flattened naming convention specified
            aug_filename = f"{name}_aug_{i:04d}{ext}"
            cv2.imwrite(os.path.join(augmented_dir, aug_filename), augmented_bgr)

    print("Bootstrapping complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap the Set Card training set by augmenting seed images.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Folder containing the 81 seed images.")
    parser.add_argument("--augmented_dir", type=str, default="data/augmented", help="Folder to save the flattened augmented dataset.")
    parser.add_argument("--augmentations", type=int, default=300, help="Number of variations per image.")
    args = parser.parse_args()

    bootstrap_dataset(args.raw_dir, args.augmented_dir, args.augmentations)
