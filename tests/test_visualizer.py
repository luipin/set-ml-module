import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt

from src.utils.visualizer import visualize_predictions
from src.models.multi_head_resnet import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_rgb_image(h=64, w=64):
    """Returns a random uint8 HxWx3 RGB array."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_tensor_image(h=64, w=64):
    """Returns a float32 CxHxW ImageNet-normalised array (as from a DataLoader)."""
    rng = np.random.default_rng(1)
    return rng.standard_normal((3, h, w)).astype(np.float32)


def _gt(color="red", shape="diamond", number="1", shading="solid"):
    return {"color": color, "shape": shape, "number": number, "shading": shading}


def _pred(color="red", shape="diamond", number="1", shading="solid"):
    return {f: {"prediction": v, "confidence": 0.9} for f, v in
            {"color": color, "shape": shape, "number": number, "shading": shading}.items()}


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid resource warnings."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVisualizePredictions:
    def test_returns_figure(self):
        imgs = [_make_rgb_image()]
        fig = visualize_predictions(imgs, [_gt()], [_pred()])
        assert isinstance(fig, plt.Figure)

    def test_mismatched_lengths_raise(self):
        imgs = [_make_rgb_image(), _make_rgb_image()]
        with pytest.raises(ValueError, match="same length"):
            visualize_predictions(imgs, [_gt()], [_pred()])

    def test_max_images_respected(self):
        n = 10
        imgs = [_make_rgb_image() for _ in range(n)]
        gts = [_gt() for _ in range(n)]
        preds = [_pred() for _ in range(n)]
        fig = visualize_predictions(imgs, gts, preds, max_images=4)
        # 4 images → 1 row of 4 → figure has 4 axes (all with imshow content)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 4

    def test_accepts_tensor_images(self):
        imgs = [_make_tensor_image()]
        fig = visualize_predictions(imgs, [_gt()], [_pred()])
        assert isinstance(fig, plt.Figure)

    def test_bad_image_shape_raises(self):
        bad = np.zeros((64, 64), dtype=np.uint8)  # 2-D, missing channel dim
        with pytest.raises(ValueError):
            visualize_predictions([bad], [_gt()], [_pred()])

    def test_correct_prediction_text_color(self):
        """When prediction matches ground truth the annotation text should be green."""
        fig = visualize_predictions(
            [_make_rgb_image()], [_gt()], [_pred()]  # all correct
        )
        ax = fig.axes[0]
        text_colors = [t.get_color() for t in ax.texts]
        assert all(c == "green" for c in text_colors)

    def test_wrong_prediction_text_color(self):
        """When prediction is wrong the annotation text should be red."""
        fig = visualize_predictions(
            [_make_rgb_image()],
            [_gt(color="red")],
            [_pred(color="green")],  # color is wrong, rest correct
        )
        ax = fig.axes[0]
        text_by_feature = ax.texts  # one text per feature
        colors = {t.get_text().split(":")[0]: t.get_color() for t in text_by_feature}
        assert colors["color"] == "red"
        for feature in ("shape", "number", "shading"):
            assert colors[feature] == "green"

    def test_all_features_annotated(self):
        fig = visualize_predictions(
            [_make_rgb_image()], [_gt()], [_pred()]
        )
        ax = fig.axes[0]
        text_labels = [t.get_text().split(":")[0] for t in ax.texts]
        for feature in FEATURE_NAMES:
            assert feature in text_labels

    def test_empty_input_returns_figure(self):
        fig = visualize_predictions([], [], [])
        assert isinstance(fig, plt.Figure)

    def test_single_image_grid(self):
        fig = visualize_predictions(
            [_make_rgb_image()], [_gt()], [_pred()], max_images=1
        )
        assert isinstance(fig, plt.Figure)

    def test_exactly_four_columns_for_large_batch(self):
        n = 12
        imgs = [_make_rgb_image() for _ in range(n)]
        gts = [_gt() for _ in range(n)]
        preds = [_pred() for _ in range(n)]
        fig = visualize_predictions(imgs, gts, preds, max_images=n)
        # 12 images → 3 rows × 4 cols → 12 axes total
        assert len(fig.axes) == 12
