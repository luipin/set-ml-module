import numpy as np
import pytest
import torch
import torchvision.models as tv_models
from unittest.mock import patch

from src.models.multi_head_resnet import MultiHeadResNet, FEATURE_NAMES
from src.models.predictor import predict, INVERSE_LABEL_MAPS


@pytest.fixture
def untrained_model():
    """Returns a randomly-initialised MultiHeadResNet on CPU (no download needed).

    Patches torchvision so the ResNet18 backbone is built with random weights
    instead of downloading ImageNet pretrained weights, making tests offline-safe.
    """
    _original_resnet18 = tv_models.resnet18
    with patch.object(tv_models, "resnet18", side_effect=lambda **kwargs: _original_resnet18(weights=None)):
        model = MultiHeadResNet(freeze_epochs=0)
    model.eval()
    return model


@pytest.fixture
def dummy_image():
    """224x224 random RGB uint8 array — stands in for a real card image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)


class TestInverseLabelMaps:
    def test_all_features_present(self):
        assert set(INVERSE_LABEL_MAPS.keys()) == set(FEATURE_NAMES)

    def test_roundtrip(self):
        """int -> label -> int should be identity for every feature."""
        from src.data.set_card_data_pipeline import LABEL_MAPS
        for feature, mapping in LABEL_MAPS.items():
            for label, idx in mapping.items():
                assert INVERSE_LABEL_MAPS[feature][idx] == label

    def test_three_classes_per_feature(self):
        for feature in FEATURE_NAMES:
            assert len(INVERSE_LABEL_MAPS[feature]) == 3


class TestPredict:
    def test_returns_all_features(self, untrained_model, dummy_image):
        result = predict(untrained_model, dummy_image)
        assert set(result.keys()) == set(FEATURE_NAMES)

    def test_output_structure(self, untrained_model, dummy_image):
        result = predict(untrained_model, dummy_image)
        for feature in FEATURE_NAMES:
            assert "prediction" in result[feature]
            assert "confidence" in result[feature]

    def test_prediction_is_valid_label(self, untrained_model, dummy_image):
        result = predict(untrained_model, dummy_image)
        valid = {f: set(INVERSE_LABEL_MAPS[f].values()) for f in FEATURE_NAMES}
        for feature in FEATURE_NAMES:
            assert result[feature]["prediction"] in valid[feature]

    def test_confidence_in_range(self, untrained_model, dummy_image):
        result = predict(untrained_model, dummy_image)
        for feature in FEATURE_NAMES:
            conf = result[feature]["confidence"]
            assert 0.0 <= conf <= 1.0

    def test_confidence_rounded_to_4dp(self, untrained_model, dummy_image):
        result = predict(untrained_model, dummy_image)
        for feature in FEATURE_NAMES:
            conf = result[feature]["confidence"]
            assert conf == round(conf, 4)

    def test_accepts_numpy_array(self, untrained_model, dummy_image):
        result = predict(untrained_model, dummy_image)
        assert isinstance(result, dict)

    def test_file_not_found_raises(self, untrained_model):
        with pytest.raises(FileNotFoundError):
            predict(untrained_model, "/nonexistent/path/card.jpg")

    def test_bad_array_shape_raises(self, untrained_model):
        bad = np.zeros((224, 224), dtype=np.uint8)  # missing channel dim
        with pytest.raises(ValueError):
            predict(untrained_model, bad)

    def test_unsupported_type_raises(self, untrained_model):
        with pytest.raises(ValueError):
            predict(untrained_model, 12345)

    def test_result_is_json_serializable(self, untrained_model, dummy_image):
        import json
        result = predict(untrained_model, dummy_image)
        # Should not raise
        json.dumps(result)
