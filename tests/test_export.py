import pytest
import torch
import torchvision.models as tv_models
from unittest.mock import patch

from src.models.multi_head_resnet import MultiHeadResNet, FEATURE_NAMES
from src.models.export import export_torchscript


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    """Randomly-initialised MultiHeadResNet — no pretrained download needed."""
    _orig = tv_models.resnet18
    with patch.object(tv_models, "resnet18", side_effect=lambda **kw: _orig(weights=None)):
        m = MultiHeadResNet(freeze_epochs=0)
    m.eval()
    return m


@pytest.fixture
def tmp_pt(tmp_path):
    """Temporary path for an export file."""
    return tmp_path / "model.pt2"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExportTorchscript:
    def test_returns_exported_program(self, model, tmp_pt):
        result = export_torchscript(model, output_path=str(tmp_pt))
        assert isinstance(result, torch.export.ExportedProgram)

    def test_file_is_created(self, model, tmp_pt):
        export_torchscript(model, output_path=str(tmp_pt))
        assert tmp_pt.exists()

    def test_parent_dirs_created(self, model, tmp_path):
        nested = tmp_path / "a" / "b" / "model.pt2"
        export_torchscript(model, output_path=str(nested))
        assert nested.exists()

    def test_exported_program_is_loadable(self, model, tmp_pt):
        export_torchscript(model, output_path=str(tmp_pt))
        loaded = torch.export.load(str(tmp_pt))
        assert isinstance(loaded, torch.export.ExportedProgram)

    def test_loaded_model_output_shape(self, model, tmp_pt):
        export_torchscript(model, output_path=str(tmp_pt))
        ep = torch.export.load(str(tmp_pt))
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = ep.module()(x)
        for feature in FEATURE_NAMES:
            assert logits[feature].shape == (1, 3)

    def test_loaded_model_has_all_features(self, model, tmp_pt):
        export_torchscript(model, output_path=str(tmp_pt))
        ep = torch.export.load(str(tmp_pt))
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = ep.module()(x)
        assert set(logits.keys()) == set(FEATURE_NAMES)

    def test_accepts_custom_example_input(self, model, tmp_pt):
        example = torch.randn(1, 3, 224, 224)
        result = export_torchscript(model, output_path=str(tmp_pt), example_input=example)
        assert isinstance(result, torch.export.ExportedProgram)

    def test_bad_example_input_shape_raises(self, model, tmp_pt):
        bad = torch.randn(1, 1, 224, 224)  # wrong channel count
        with pytest.raises(ValueError, match="224, 224"):
            export_torchscript(model, output_path=str(tmp_pt), example_input=bad)

    def test_output_matches_original_model(self, model, tmp_pt):
        """Exported model output must be numerically identical to the original."""
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            orig_logits = model(x)

        export_torchscript(model, output_path=str(tmp_pt))
        ep = torch.export.load(str(tmp_pt))
        with torch.no_grad():
            exported_logits = ep.module()(x)

        for feature in FEATURE_NAMES:
            assert torch.allclose(orig_logits[feature], exported_logits[feature])
