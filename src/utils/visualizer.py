import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.models.multi_head_resnet import FEATURE_NAMES


def visualize_predictions(images, ground_truths, predictions, max_images: int = 16) -> plt.Figure:
    """Display a grid of Set card images annotated with ground truth and predictions.

    Each cell in the grid shows one card image. Below the image, each of the
    four features is printed on its own line:

    * **Green** text — the model predicted this feature correctly.
    * **Red** text   — the model predicted this feature incorrectly; both the
      ground-truth value and the (wrong) prediction are shown so errors are
      easy to diagnose.

    The function is designed to be called inside a Jupyter notebook cell, where
    the returned ``Figure`` is displayed automatically, but it can also be used
    in scripts by calling ``plt.show()`` or ``fig.savefig(...)`` on the result.

    Args:
        images: Sequence of card images to display. Each element may be:

            * A ``numpy.ndarray`` of shape ``(H, W, 3)`` with uint8 pixel
              values in RGB order — displayed as-is.
            * A ``numpy.ndarray`` of shape ``(3, H, W)`` with float32 values
              (i.e., a normalised PyTorch tensor that has been moved to CPU and
              converted with ``.numpy()``). The function will transpose and
              de-normalise it using ImageNet statistics so it renders correctly.

        ground_truths: Sequence of ground-truth label dicts, one per image.
            Each dict must map every feature name in ``FEATURE_NAMES`` to a
            human-readable string, e.g.::

                {"color": "red", "shape": "diamond", "number": "1", "shading": "solid"}

        predictions: Sequence of prediction dicts in the same format returned
            by ``src.models.predictor.predict()``, i.e.::

                {"color": {"prediction": "red", "confidence": 0.97}, ...}

            Only the ``"prediction"`` key is used for colour-coding; confidence
            values are ignored by this function.

        max_images: Maximum number of images to display. If the input sequences
            are longer than this value, only the first ``max_images`` elements
            are shown. Defaults to 16.

    Returns:
        matplotlib.figure.Figure: The figure object containing the grid. In a
        Jupyter notebook the figure is rendered automatically when it is the
        last expression in a cell. In a script, call ``plt.show()`` or
        ``fig.savefig("debug.png")`` to display or save it.

    Raises:
        ValueError: If ``images``, ``ground_truths``, and ``predictions`` do
            not all have the same length, or if any image array has an
            unrecognised shape.

    Example::

        from src.models.predictor import predict
        from src.utils.visualizer import visualize_predictions

        preds = [predict(model, img) for img in images]
        fig = visualize_predictions(images, ground_truths, preds, max_images=8)
        fig.savefig("debug_grid.png", bbox_inches="tight")
    """
    if not (len(images) == len(ground_truths) == len(predictions)):
        raise ValueError(
            f"images ({len(images)}), ground_truths ({len(ground_truths)}), and "
            f"predictions ({len(predictions)}) must all have the same length."
        )

    n = min(len(images), max_images)
    images = list(images)[:n]
    ground_truths = list(ground_truths)[:n]
    predictions = list(predictions)[:n]

    # Grid layout: aim for roughly square, at most 4 columns.
    n_cols = min(n, 4) if n > 0 else 1
    n_rows = math.ceil(n / n_cols) if n > 0 else 1

    # Each cell is tall enough for the image + 4 feature lines of text.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.2, n_rows * 4.2),
        squeeze=False,
    )

    for i in range(n_rows * n_cols):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        ax.axis("off")

        if i >= n:
            continue

        img = _prepare_image(images[i], i)
        ax.imshow(img)

        # Build the annotation text, one line per feature.
        lines = []
        colors = []
        for feature in FEATURE_NAMES:
            gt_label = str(ground_truths[i][feature])
            pred_entry = predictions[i][feature]
            pred_label = (
                pred_entry["prediction"]
                if isinstance(pred_entry, dict)
                else str(pred_entry)
            )
            correct = gt_label == pred_label
            if correct:
                lines.append(f"{feature}: {gt_label}")
                colors.append("green")
            else:
                lines.append(f"{feature}: {gt_label} \u2192 {pred_label}")
                colors.append("red")

        # Render each feature line individually so we can colour them.
        y_start = -0.04          # just above the image bottom edge (axes coords)
        line_height = 0.115
        for k, (line, color) in enumerate(zip(lines, colors)):
            ax.text(
                0.5,
                y_start - k * line_height,
                line,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                color=color,
                fontweight="bold" if color == "red" else "normal",
            )

    # Legend
    legend_handles = [
        mpatches.Patch(color="green", label="Correct"),
        mpatches.Patch(color="red",   label="Incorrect (gt \u2192 pred)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# ImageNet statistics used during training — needed to de-normalise tensors.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _prepare_image(img: np.ndarray, idx: int) -> np.ndarray:
    """Converts an image array to a uint8 HxWx3 RGB array suitable for imshow.

    Args:
        img: Either a uint8 (H, W, 3) RGB array or a float32 (3, H, W)
            ImageNet-normalised tensor array.
        idx: Index of the image in the batch (used in error messages only).

    Returns:
        numpy.ndarray: uint8 array of shape (H, W, 3) ready for ``imshow``.

    Raises:
        ValueError: If the array shape is not recognised.
    """
    if img.ndim == 3 and img.shape[2] == 3:
        # Already HxWx3 — assume uint8 RGB, return as-is.
        return img.astype(np.uint8)

    if img.ndim == 3 and img.shape[0] == 3:
        # CxHxW float32 (normalised tensor). Transpose and de-normalise.
        img = img.transpose(1, 2, 0)                          # -> HxWxC
        img = img * _IMAGENET_STD + _IMAGENET_MEAN            # de-normalise
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    raise ValueError(
        f"Image at index {idx} has unrecognised shape {img.shape}. "
        "Expected (H, W, 3) uint8 or (3, H, W) float32."
    )
