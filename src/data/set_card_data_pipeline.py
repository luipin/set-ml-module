import os
import glob
import cv2
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.utils.augmentations import get_train_transforms, get_val_transforms

# Label mappings (Feature -> Value -> Integer)
LABEL_MAPS = {
    'color': {'red': 0, 'green': 1, 'purple': 2},
    'shape': {'diamond': 0, 'squiggle': 1, 'oval': 2},
    'number': {'1': 0, '2': 1, '3': 2},
    'shading': {'solid': 0, 'striped': 1, 'open': 2}
}

# Filename aliases: some seed images use alternative spellings that map to a
# canonical label. e.g. 'empty' and 'open' refer to the same Set card shading.
_LABEL_ALIASES = {
    'shading': {'empty': 'open'},
}

class SetCardDataset(Dataset):
    """
    Custom PyTorch Dataset for Set Cards.

    Parses labels from filenames: {color}_{shape}_{number}_{shading}[_optional_suffix].jpg
    """
    def __init__(self, image_paths, transform=None):
        """
        Initializes the dataset with image paths and transformations.

        Args:
            image_paths (list): List of paths to the images.
            transform (callable, optional): Albumentations transform pipeline.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Loads an image, parses labels from the filename, and applies transforms.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image_tensor, labels_dict) where labels_dict contains 4 tensors.

        Raises:
            ValueError: If the image cannot be read or the filename is malformed.
            KeyError: If a parsed label is not found in LABEL_MAPS.
        """
        img_path = self.image_paths[idx]
        
        # Read image using OpenCV (returns BGR, so we convert to RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to read image at {img_path}. File might be missing or corrupted.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parse filename for labels (e.g., 'red_diamond_1_solid_aug_001.jpg')
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        
        if len(parts) < 4:
            raise ValueError(
                f"Filename '{filename}' does not match expected format: "
                "{color}_{shape}_{number}_{shading}[_...].jpg"
            )

        try:
            color_str   = parts[0].lower()
            shape_str   = parts[1].lower()
            number_str  = parts[2].lower()
            shading_str = parts[3].split('.')[0].lower()  # Drop extension or suffix
            # Normalise any filename alias to the canonical label name
            shading_str = _LABEL_ALIASES['shading'].get(shading_str, shading_str)

            labels = {
                'color': torch.tensor(LABEL_MAPS['color'][color_str], dtype=torch.long),
                'shape': torch.tensor(LABEL_MAPS['shape'][shape_str], dtype=torch.long),
                'number': torch.tensor(LABEL_MAPS['number'][number_str], dtype=torch.long),
                'shading': torch.tensor(LABEL_MAPS['shading'][shading_str], dtype=torch.long)
            }
        except KeyError as e:
            raise KeyError(
                f"Invalid label '{e.args[0]}' found in filename '{filename}'. "
                f"Expected values from {list(LABEL_MAPS.keys())}"
            )

        # Apply Albumentations Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, labels

class SetCardDataModule(pl.LightningDataModule):
    """
    LightningDataModule handling data loading and augmentation for the Set Card Classifier.
    """
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, dynamic_multiplier: int = 1):
        """
        Initializes the DataModule.

        Args:
            data_dir: Path to directory containing images (raw or augmented).
            batch_size: DataLoader batch size.
            num_workers: Number of workers for DataLoaders.
            dynamic_multiplier: Artificially repeat the dataset N times per epoch. Useful 
                                if using only the 81 seed images and dynamically augmenting.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dynamic_multiplier = dynamic_multiplier
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """
        Prepares training and validation datasets by splitting available images.

        Args:
            stage (str, optional): Current stage (fit/test/predict). Defaults to None.

        Raises:
            FileNotFoundError: If no valid image files are found in data_dir.
        """
        # Discover all JPGs and PNGs
        all_images = glob.glob(os.path.join(self.data_dir, '*.jpg')) + \
                     glob.glob(os.path.join(self.data_dir, '*.png'))
                     
        if len(all_images) == 0:
            raise FileNotFoundError(
                f"No images found in {self.data_dir}. Ensure the directory exists "
                "and contains .jpg or .png files with valid naming conventions."
            )

        # Perform 80/20 train-test split
        train_paths, val_paths = train_test_split(all_images, test_size=0.2, random_state=42)

        # Apply dynamic multiplier to artificially expand the training set size per epoch 
        # (e.g., 81 seed images * 300 multiplier = 24,300 iterations per epoch)
        train_paths = train_paths * self.dynamic_multiplier

        self.train_dataset = SetCardDataset(train_paths, transform=get_train_transforms())
        self.val_dataset = SetCardDataset(val_paths, transform=get_val_transforms())

    def train_dataloader(self):
        """Returns the training DataLoader."""
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        """Returns the validation DataLoader."""
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
