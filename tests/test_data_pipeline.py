import math
import cv2
import numpy as np
import pytest
import torch
from pathlib import Path

from src.data.set_card_data_pipeline import (
    SetCardDataset,
    SetCardDataModule,
    LABEL_MAPS,
)
from src.utils.augmentations import get_val_transforms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_card_image(directory: Path, filename: str, size: int = 32) -> Path:
    """Write a small solid-colour JPEG to directory/filename and return the path."""
    img = np.full((size, size, 3), fill_value=128, dtype=np.uint8)
    path = directory / filename
    cv2.imwrite(str(path), img)
    return path


def _make_seed_images(directory: Path, n: int = 10) -> list[Path]:
    """Write n valid card images (cycling through known combos) and return paths."""
    combos = [
        ("red", "diamond", "1", "solid"),
        ("green", "squiggle", "2", "striped"),
        ("purple", "oval", "3", "open"),
        ("red", "oval", "2", "open"),
        ("green", "diamond", "3", "solid"),
        ("purple", "squiggle", "1", "striped"),
        ("red", "squiggle", "3", "open"),
        ("green", "oval", "1", "solid"),
        ("purple", "diamond", "2", "open"),
        ("red", "diamond", "2", "striped"),
    ]
    paths = []
    for i in range(n):
        color, shape, number, shading = combos[i % len(combos)]
        fname = f"{color}_{shape}_{number}_{shading}.jpg"
        paths.append(_write_card_image(directory, fname))
    return paths


# ---------------------------------------------------------------------------
# LABEL_MAPS sanity checks
# ---------------------------------------------------------------------------

class TestLabelMaps:
    def test_all_features_present(self):
        assert set(LABEL_MAPS.keys()) == {"color", "shape", "number", "shading"}

    def test_three_classes_per_feature(self):
        for feature, mapping in LABEL_MAPS.items():
            assert len(mapping) == 3, f"{feature} should have 3 classes"

    def test_indices_are_zero_one_two(self):
        for feature, mapping in LABEL_MAPS.items():
            assert set(mapping.values()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# SetCardDataset — label parsing
# ---------------------------------------------------------------------------

class TestSetCardDatasetLabelParsing:
    def test_valid_filename_parses_correctly(self, tmp_path):
        path = _write_card_image(tmp_path, "red_diamond_1_solid.jpg")
        ds = SetCardDataset([str(path)], transform=get_val_transforms())
        _, labels = ds[0]
        assert labels["color"].item()   == LABEL_MAPS["color"]["red"]
        assert labels["shape"].item()   == LABEL_MAPS["shape"]["diamond"]
        assert labels["number"].item()  == LABEL_MAPS["number"]["1"]
        assert labels["shading"].item() == LABEL_MAPS["shading"]["solid"]

    def test_all_colors_parse(self, tmp_path):
        for color, idx in LABEL_MAPS["color"].items():
            path = _write_card_image(tmp_path, f"{color}_diamond_1_solid.jpg")
            ds = SetCardDataset([str(path)], transform=get_val_transforms())
            _, labels = ds[0]
            assert labels["color"].item() == idx

    def test_all_shapes_parse(self, tmp_path):
        for shape, idx in LABEL_MAPS["shape"].items():
            path = _write_card_image(tmp_path, f"red_{shape}_1_solid.jpg")
            ds = SetCardDataset([str(path)], transform=get_val_transforms())
            _, labels = ds[0]
            assert labels["shape"].item() == idx

    def test_all_numbers_parse(self, tmp_path):
        for number, idx in LABEL_MAPS["number"].items():
            path = _write_card_image(tmp_path, f"red_diamond_{number}_solid.jpg")
            ds = SetCardDataset([str(path)], transform=get_val_transforms())
            _, labels = ds[0]
            assert labels["number"].item() == idx

    def test_all_shadings_parse(self, tmp_path):
        for shading, idx in LABEL_MAPS["shading"].items():
            path = _write_card_image(tmp_path, f"red_diamond_1_{shading}.jpg")
            ds = SetCardDataset([str(path)], transform=get_val_transforms())
            _, labels = ds[0]
            assert labels["shading"].item() == idx

    def test_augmented_filename_with_suffix_parses(self, tmp_path):
        """Filenames like red_diamond_1_solid_aug_001.jpg should still parse."""
        path = _write_card_image(tmp_path, "red_diamond_1_solid_aug_001.jpg")
        ds = SetCardDataset([str(path)], transform=get_val_transforms())
        _, labels = ds[0]
        assert labels["color"].item() == LABEL_MAPS["color"]["red"]
        assert labels["shading"].item() == LABEL_MAPS["shading"]["solid"]

    def test_labels_are_long_tensors(self, tmp_path):
        path = _write_card_image(tmp_path, "red_diamond_1_solid.jpg")
        ds = SetCardDataset([str(path)], transform=get_val_transforms())
        _, labels = ds[0]
        for feature in ("color", "shape", "number", "shading"):
            assert labels[feature].dtype == torch.long


# ---------------------------------------------------------------------------
# SetCardDataset — error handling
# ---------------------------------------------------------------------------

class TestSetCardDatasetErrors:
    def test_malformed_filename_raises_value_error(self, tmp_path):
        """Filename with fewer than 4 underscore-separated parts should raise."""
        path = _write_card_image(tmp_path, "bad_name.jpg")
        ds = SetCardDataset([str(path)])
        with pytest.raises(ValueError, match="format"):
            ds[0]

    def test_invalid_color_raises_key_error(self, tmp_path):
        path = _write_card_image(tmp_path, "blue_diamond_1_solid.jpg")
        ds = SetCardDataset([str(path)])
        with pytest.raises(KeyError):
            ds[0]

    def test_invalid_shape_raises_key_error(self, tmp_path):
        path = _write_card_image(tmp_path, "red_circle_1_solid.jpg")
        ds = SetCardDataset([str(path)])
        with pytest.raises(KeyError):
            ds[0]

    def test_missing_image_raises_value_error(self, tmp_path):
        ds = SetCardDataset([str(tmp_path / "nonexistent.jpg")])
        with pytest.raises(ValueError, match="Failed to read"):
            ds[0]


# ---------------------------------------------------------------------------
# SetCardDataset — length and output shapes
# ---------------------------------------------------------------------------

class TestSetCardDatasetOutputs:
    def test_len_returns_correct_count(self, tmp_path):
        paths = _make_seed_images(tmp_path, n=7)
        ds = SetCardDataset([str(p) for p in paths])
        assert len(ds) == 7

    def test_image_tensor_shape(self, tmp_path):
        path = _write_card_image(tmp_path, "red_diamond_1_solid.jpg")
        ds = SetCardDataset([str(path)], transform=get_val_transforms())
        image, _ = ds[0]
        # get_val_transforms resizes to 224x224 and converts to tensor
        assert image.shape == (3, 224, 224)

    def test_image_tensor_dtype(self, tmp_path):
        path = _write_card_image(tmp_path, "red_diamond_1_solid.jpg")
        ds = SetCardDataset([str(path)], transform=get_val_transforms())
        image, _ = ds[0]
        assert image.dtype == torch.float32

    def test_no_transform_returns_numpy(self, tmp_path):
        """Without a transform, __getitem__ returns a raw HxWx3 numpy array."""
        path = _write_card_image(tmp_path, "red_diamond_1_solid.jpg")
        ds = SetCardDataset([str(path)], transform=None)
        image, _ = ds[0]
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3 and image.shape[2] == 3


# ---------------------------------------------------------------------------
# SetCardDataModule
# ---------------------------------------------------------------------------

class TestSetCardDataModule:
    def test_setup_raises_when_dir_empty(self, tmp_path):
        dm = SetCardDataModule(data_dir=str(tmp_path), batch_size=4, num_workers=0)
        with pytest.raises(FileNotFoundError):
            dm.setup()

    def test_setup_creates_train_and_val_datasets(self, tmp_path):
        _make_seed_images(tmp_path, n=10)
        dm = SetCardDataModule(data_dir=str(tmp_path), batch_size=4, num_workers=0)
        dm.setup()
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None

    def test_80_20_split(self, tmp_path):
        n = 10
        _make_seed_images(tmp_path, n=n)
        dm = SetCardDataModule(data_dir=str(tmp_path), batch_size=4, num_workers=0)
        dm.setup()
        expected_val = math.floor(n * 0.2)
        expected_train = n - expected_val
        assert len(dm.val_dataset) == expected_val
        assert len(dm.train_dataset) == expected_train

    def test_dynamic_multiplier_expands_train_set(self, tmp_path):
        n = 10
        _make_seed_images(tmp_path, n=n)
        dm = SetCardDataModule(
            data_dir=str(tmp_path), batch_size=4, num_workers=0, dynamic_multiplier=3
        )
        dm.setup()
        expected_val = math.floor(n * 0.2)
        expected_train_base = n - expected_val
        assert len(dm.train_dataset) == expected_train_base * 3

    def test_train_dataloader_batch_shapes(self, tmp_path):
        _make_seed_images(tmp_path, n=10)
        dm = SetCardDataModule(data_dir=str(tmp_path), batch_size=4, num_workers=0)
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        images, labels = batch
        assert images.shape[1:] == (3, 224, 224)
        for feature in ("color", "shape", "number", "shading"):
            assert labels[feature].shape == (images.shape[0],)
            assert labels[feature].dtype == torch.long

    def test_val_dataloader_batch_shapes(self, tmp_path):
        _make_seed_images(tmp_path, n=10)
        dm = SetCardDataModule(data_dir=str(tmp_path), batch_size=4, num_workers=0)
        dm.setup()
        batch = next(iter(dm.val_dataloader()))
        images, labels = batch
        assert images.shape[1:] == (3, 224, 224)
        for feature in ("color", "shape", "number", "shading"):
            assert labels[feature].shape == (images.shape[0],)
