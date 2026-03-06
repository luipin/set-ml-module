import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from src.data.set_card_data_pipeline import LABEL_MAPS
from src.models.multi_head_resnet import MultiHeadResNet, FEATURE_NAMES
from src.utils.augmentations import get_val_transforms

# Invert LABEL_MAPS so we can convert integer predictions back to human-readable strings.
# e.g. INVERSE_LABEL_MAPS['color'][0] -> 'red'
INVERSE_LABEL_MAPS = {
    feature: {idx: label for label, idx in mapping.items()}
    for feature, mapping in LABEL_MAPS.items()
}


def predict(model: MultiHeadResNet, image) -> dict:
    """Runs inference on a single Set card image and returns human-readable predictions.

    Applies the same validation preprocessing pipeline used during training
    (resize to 224×224, ImageNet normalization), then runs a forward pass and
    converts the raw logits to softmax probabilities. The highest-probability
    class is reported as the prediction, and its probability is the confidence.

    Args:
        model: A trained ``MultiHeadResNet`` instance. The model is automatically
            moved to eval mode and no gradients are computed during inference.
        image: The card image to classify. Accepted formats:

            * ``str`` or ``pathlib.Path`` — path to a JPEG/PNG file on disk.
            * ``numpy.ndarray`` — an HxWx3 RGB or BGR uint8 array. If the
              array appears to be BGR (as returned by ``cv2.imread``), convert
              it to RGB before passing it in, or pass a file path instead.

    Returns:
        A JSON-serializable dict with one key per Set card feature. Each value
        is itself a dict with two fields::

            {
                "color":   {"prediction": "red",     "confidence": 0.97},
                "shape":   {"prediction": "diamond", "confidence": 0.88},
                "number":  {"prediction": "2",       "confidence": 0.95},
                "shading": {"prediction": "solid",   "confidence": 0.91},
            }

        ``prediction`` is the human-readable label string.
        ``confidence`` is a float in [0, 1] rounded to four decimal places.

    Raises:
        FileNotFoundError: If ``image`` is a path and the file does not exist.
        ValueError: If ``image`` is a numpy array with an unexpected shape, or
            if the file at the given path cannot be decoded by OpenCV.

    Example::

        model = MultiHeadResNet.load_from_checkpoint("checkpoints/best.ckpt")
        result = predict(model, "data/raw/red_diamond_1_solid.jpg")
        print(result["color"]["prediction"])   # "red"
        print(result["color"]["confidence"])   # 0.9734
    """
    # --- 1. Load and preprocess ---
    transform = get_val_transforms()
    tensor = transform(image=_load_image(image))["image"]  # shape: (3, 224, 224)
    batch = tensor.unsqueeze(0)  # shape: (1, 3, 224, 224)

    # --- 3. Inference ---
    # Determine the device the model lives on and move the input there.
    device = next(model.parameters()).device
    batch = batch.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(batch)  # dict[str, Tensor(1, 3)]

    # --- 4. Convert logits -> probabilities -> result dict ---
    result = {}
    for feature in FEATURE_NAMES:
        probs = F.softmax(logits[feature], dim=1)  # (1, 3)
        confidence, class_idx = probs.max(dim=1)
        label = INVERSE_LABEL_MAPS[feature][class_idx.item()]
        result[feature] = {
            "prediction": label,
            "confidence": round(confidence.item(), 4),
        }

    return result


def _load_image(image) -> np.ndarray:
    """Loads and validates a single image into an HxWx3 RGB numpy array.

    Args:
        image: A file path (str/Path) or an HxWx3 numpy array.

    Returns:
        HxWx3 RGB uint8 numpy array.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the image cannot be decoded or has wrong shape.
    """
    if isinstance(image, (str, Path)):
        path = str(image)
        if not Path(path).exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise ValueError(f"OpenCV could not decode image at: {path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected a 3-channel HxWx3 array, got shape {image.shape}."
            )
        return image
    else:
        raise ValueError(
            f"Unsupported image type: {type(image)}. "
            "Pass a file path (str/Path) or a numpy HxWx3 RGB array."
        )


def predict_batch(model: MultiHeadResNet, images: list) -> list[dict]:
    """Runs inference on a batch of Set card images in a single forward pass.

    More efficient than calling ``predict()`` in a loop because all images
    share one GPU/CPU kernel launch and one matrix multiplication per head.
    The speedup is most pronounced on GPU where kernel launch overhead
    dominates single-image latency.

    Args:
        model: A trained ``MultiHeadResNet`` instance.
        images: A list of images. Each element may be a file path (str/Path)
            or an HxWx3 RGB numpy array — the same formats accepted by
            ``predict()``.

    Returns:
        A list of prediction dicts, one per input image, in the same order as
        ``images``. Each dict has the same structure as ``predict()``::

            [
                {"color": {"prediction": "red", "confidence": 0.97}, ...},
                {"color": {"prediction": "green", "confidence": 0.91}, ...},
            ]

    Raises:
        ValueError: If ``images`` is empty.
        FileNotFoundError: If any path in ``images`` does not exist.

    Example::

        paths = glob.glob("data/raw/*.jpg")
        results = predict_batch(model, paths)
        for path, result in zip(paths, results):
            print(path, result["color"]["prediction"])
    """
    if not images:
        raise ValueError("images list must not be empty.")

    transform = get_val_transforms()
    device = next(model.parameters()).device

    # Preprocess all images and stack into a single (B, 3, 224, 224) tensor.
    tensors = [transform(image=_load_image(img))["image"] for img in images]
    batch = torch.stack(tensors).to(device)  # (B, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        logits = model(batch)  # dict[str, Tensor(B, 3)]

    # Unpack batch dimension: produce one result dict per image.
    B = batch.shape[0]
    results = []
    for i in range(B):
        result = {}
        for feature in FEATURE_NAMES:
            probs = F.softmax(logits[feature][i], dim=0)  # (3,)
            confidence, class_idx = probs.max(dim=0)
            label = INVERSE_LABEL_MAPS[feature][class_idx.item()]
            result[feature] = {
                "prediction": label,
                "confidence": round(confidence.item(), 4),
            }
        results.append(result)

    return results
