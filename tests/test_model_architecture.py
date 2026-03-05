import pytest
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from unittest.mock import patch

from src.models.multi_head_resnet import MultiHeadResNet, FEATURE_NAMES, NUM_CLASSES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model(freeze_epochs=0):
    """Randomly-initialised MultiHeadResNet — no pretrained download needed."""
    _orig = tv_models.resnet18
    with patch.object(tv_models, "resnet18", side_effect=lambda **kw: _orig(weights=None)):
        return MultiHeadResNet(freeze_epochs=freeze_epochs)


def _make_batch(batch_size=4):
    """Random image batch of shape (B, 3, 224, 224)."""
    return torch.randn(batch_size, 3, 224, 224)


def _make_labels(batch_size=4):
    """Random ground-truth label dict with long tensors of shape (B,)."""
    return {feature: torch.randint(0, NUM_CLASSES, (batch_size,)) for feature in FEATURE_NAMES}


@pytest.fixture
def model():
    return _make_model(freeze_epochs=0)


@pytest.fixture
def frozen_model():
    return _make_model(freeze_epochs=5)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_output_is_dict(self, model):
        logits = model(_make_batch())
        assert isinstance(logits, dict)

    def test_output_keys(self, model):
        logits = model(_make_batch())
        assert set(logits.keys()) == set(FEATURE_NAMES)

    def test_output_shape_per_head(self, model):
        batch_size = 4
        logits = model(_make_batch(batch_size))
        for feature in FEATURE_NAMES:
            assert logits[feature].shape == (batch_size, NUM_CLASSES), \
                f"{feature} head: expected ({batch_size}, {NUM_CLASSES}), got {logits[feature].shape}"

    def test_output_shape_batch_size_1(self, model):
        logits = model(_make_batch(batch_size=1))
        for feature in FEATURE_NAMES:
            assert logits[feature].shape == (1, NUM_CLASSES)

    def test_output_shape_batch_size_8(self, model):
        logits = model(_make_batch(batch_size=8))
        for feature in FEATURE_NAMES:
            assert logits[feature].shape == (8, NUM_CLASSES)

    def test_output_dtype_is_float(self, model):
        logits = model(_make_batch())
        for feature in FEATURE_NAMES:
            assert logits[feature].dtype == torch.float32


# ---------------------------------------------------------------------------
# Architecture structure
# ---------------------------------------------------------------------------

class TestArchitecture:
    def test_has_backbone(self, model):
        assert hasattr(model, "backbone")

    def test_has_heads(self, model):
        assert hasattr(model, "heads")

    def test_head_count(self, model):
        assert len(model.heads) == len(FEATURE_NAMES)

    def test_head_keys_match_feature_names(self, model):
        assert set(model.heads.keys()) == set(FEATURE_NAMES)

    def test_backbone_fc_is_identity(self, model):
        import torch.nn as nn
        assert isinstance(model.backbone.fc, nn.Identity)

    def test_num_classes_constant(self):
        assert NUM_CLASSES == 3


# ---------------------------------------------------------------------------
# Backbone freezing / unfreezing
# ---------------------------------------------------------------------------

class TestFreezing:
    def test_backbone_frozen_at_init(self, frozen_model):
        for param in frozen_model.backbone.parameters():
            assert not param.requires_grad, "Backbone should be frozen at init"

    def test_heads_not_frozen_at_init(self, frozen_model):
        for param in frozen_model.heads.parameters():
            assert param.requires_grad, "Heads should always require grad"

    def test_no_freezing_when_freeze_epochs_zero(self, model):
        for param in model.backbone.parameters():
            assert param.requires_grad, "Backbone should not be frozen when freeze_epochs=0"

    def test_unfreeze_backbone_enables_grad(self, frozen_model):
        frozen_model._unfreeze_backbone()
        for param in frozen_model.backbone.parameters():
            assert param.requires_grad, "Backbone should require grad after unfreezing"

    def test_freeze_then_unfreeze(self):
        m = _make_model(freeze_epochs=0)
        m._freeze_backbone()
        for param in m.backbone.parameters():
            assert not param.requires_grad
        m._unfreeze_backbone()
        for param in m.backbone.parameters():
            assert param.requires_grad


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

class TestLossComputation:
    def test_combined_loss_is_scalar(self, model):
        batch = _make_batch()
        labels = _make_labels()
        logits = model(batch)
        loss = sum(F.cross_entropy(logits[f], labels[f]) for f in FEATURE_NAMES)
        assert loss.shape == torch.Size([])

    def test_loss_is_positive(self, model):
        batch = _make_batch()
        labels = _make_labels()
        logits = model(batch)
        loss = sum(F.cross_entropy(logits[f], labels[f]) for f in FEATURE_NAMES)
        assert loss.item() > 0

    def test_loss_is_differentiable(self, model):
        batch = _make_batch()
        labels = _make_labels()
        logits = model(batch)
        loss = sum(F.cross_entropy(logits[f], labels[f]) for f in FEATURE_NAMES)
        loss.backward()
        # At least the head parameters should have gradients
        for param in model.heads.parameters():
            assert param.grad is not None

    def test_frozen_backbone_gets_no_grad(self, frozen_model):
        batch = _make_batch()
        labels = _make_labels()
        logits = frozen_model(batch)
        loss = sum(F.cross_entropy(logits[f], labels[f]) for f in FEATURE_NAMES)
        loss.backward()
        for param in frozen_model.backbone.parameters():
            assert param.grad is None, "Frozen backbone should not accumulate gradients"


# ---------------------------------------------------------------------------
# Hyperparameter saving
# ---------------------------------------------------------------------------

class TestHyperparameters:
    def test_hparams_saved(self, model):
        assert hasattr(model, "hparams")
        assert "lr" in model.hparams
        assert "weight_decay" in model.hparams

    def test_default_lr(self, model):
        assert model.hparams.lr == pytest.approx(3e-4)

    def test_default_weight_decay(self, model):
        assert model.hparams.weight_decay == pytest.approx(1e-2)
