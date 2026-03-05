"""Generates notebooks/set_card_classifier_main.ipynb from scratch."""
import json
import os


def md(cell_id, source):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source}


def code(cell_id, source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ---------------------------------------------------------------------------
# Cell content
# ---------------------------------------------------------------------------

title = """\
# Set Card Classifier — End-to-End Pipeline

Multi-Task CNN that simultaneously predicts four features of a Set card \
(color, shape, number, shading) from a single image. The backbone is a \
pretrained ResNet18 with four parallel classification heads — one per feature.

Run all cells from top to bottom. In Google Colab, start with the Setup \
section to install dependencies and clone the repository.\
"""

s1_md = """\
## Section 1: Setup & Imports

Install dependencies and import all project modules. The pip-install cell \
only needs to run once in a fresh Colab environment — skip it if running \
locally with the project virtualenv active.\
"""

s1_install = """\
# Run only in Google Colab — skip if running locally with the project venv active
import sys
if "google.colab" in sys.modules:
    !pip install pytorch-lightning torchmetrics albumentations opencv-python-headless --quiet\
"""

s1_path = """\
import os
import sys

# In Google Colab: clone the repo and change into it
if "google.colab" in sys.modules:
    !git clone https://github.com/luipin/set-ml-module.git --quiet
    os.chdir("set-ml-module")

# Add repo root to sys.path so the 'src' package is importable
repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)\
"""

s1_imports = """\
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.data.set_card_data_pipeline import SetCardDataModule, LABEL_MAPS
from src.models.multi_head_resnet import MultiHeadResNet, FEATURE_NAMES
from src.models.predictor import predict, INVERSE_LABEL_MAPS
from src.utils.visualizer import visualize_predictions
from src.models.export import export_model

# Detect the best available hardware accelerator
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"PyTorch     : {torch.__version__}")
print(f"Accelerator : {DEVICE}")\
"""

s2_md = """\
## Section 2: Data Bootstrapping

Two data modes are supported:

| Mode | Data directory | `dynamic_multiplier` | Benefit |
|------|----------------|----------------------|---------|
| **Online** | `data/raw/` (81 seed images) | 300 | Zero pre-processing; every epoch sees freshly augmented samples |
| **Offline** | `data/augmented/` (~24k images) | 1 | Consistent augmentations; useful for reproducibility |

The cell below auto-detects which data is available and configures \
`DATA_DIR` accordingly. To generate the offline dataset, run \
`python src/data/bootstrap_dataset.py` once before training.\
"""

s2_bootstrap = """\
AUGMENTED_DIR = "data/augmented"
RAW_DIR = "data/raw"

augmented_images = glob.glob(os.path.join(AUGMENTED_DIR, "*.jpg"))
raw_images = glob.glob(os.path.join(RAW_DIR, "*.jpg"))

if len(augmented_images) > 100:
    DATA_DIR = AUGMENTED_DIR
    DYNAMIC_MULTIPLIER = 1
    print(f"Offline mode: {len(augmented_images):,} augmented images found.")
elif len(raw_images) > 0:
    DATA_DIR = RAW_DIR
    DYNAMIC_MULTIPLIER = 300
    print(
        f"Online mode: {len(raw_images)} seed images — "
        f"each epoch sees {len(raw_images) * DYNAMIC_MULTIPLIER:,} augmented samples."
    )
else:
    raise FileNotFoundError(
        f"No images found in {RAW_DIR!r} or {AUGMENTED_DIR!r}. "
        "Ensure data/raw/ contains the 81 seed images."
    )\
"""

s3_md = """\
## Section 3: Data Exploration

Before training, confirm the dataset is balanced — each of the three classes \
per feature should appear equally often. An imbalanced dataset would bias the \
model toward the majority class.\
"""

s3_dist = """\
from collections import Counter

all_raw = glob.glob(os.path.join(RAW_DIR, "*.jpg"))
counters = {feature: Counter() for feature in FEATURE_NAMES}

for path in all_raw:
    parts = os.path.basename(path).split("_")
    if len(parts) >= 4:
        counters["color"][parts[0]] += 1
        counters["shape"][parts[1]] += 1
        counters["number"][parts[2]] += 1
        counters["shading"][parts[3].split(".")[0]] += 1

fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, feature in zip(axes, FEATURE_NAMES):
    labels_list = list(counters[feature].keys())
    values_list = list(counters[feature].values())
    ax.bar(labels_list, values_list, color="steelblue")
    ax.set_title(feature.capitalize())
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(values_list) + 5)
plt.suptitle(f"Label Distribution — {len(all_raw)} seed images", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()\
"""

s3_samples = """\
import cv2

sample_paths = random.sample(all_raw, min(8, len(all_raw)))
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, path in zip(axes.flat, sample_paths):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    parts = os.path.basename(path).split("_")
    ax.set_title(f"{parts[0]} {parts[1]}\\n{parts[2]} {parts[3].split('.')[0]}", fontsize=9)
    ax.axis("off")
plt.suptitle("Sample Seed Images", fontsize=13)
plt.tight_layout()
plt.show()\
"""

s4_md = """\
## Section 4: DataModule Setup

`SetCardDataModule` wraps the dataset into Lightning's `LightningDataModule`, \
handling the 80/20 train-val split and DataLoader configuration automatically. \
The `dynamic_multiplier` parameter repeats the seed image path list N times so \
the online augmentation pipeline produces enough iterations per epoch.\
"""

s4_dm = """\
NUM_WORKERS = min(4, os.cpu_count() or 2)
BATCH_SIZE = 32

dm = SetCardDataModule(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    dynamic_multiplier=DYNAMIC_MULTIPLIER,
)
dm.setup()

print(f"Data directory     : {DATA_DIR}")
print(f"Training samples   : {len(dm.train_dataset):,}")
print(f"Validation samples : {len(dm.val_dataset):,}")
print(f"Batch size         : {BATCH_SIZE}")\
"""

s4_batch = """\
# Peek at one training batch to confirm tensor shapes and label types
_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])


def denorm(tensor):
    # Convert a normalised CHW float tensor to a displayable HWC uint8 array
    img = tensor.permute(1, 2, 0).numpy()
    img = img * _STD + _MEAN
    return np.clip(img * 255, 0, 255).astype(np.uint8)


sample_batch = next(iter(dm.train_dataloader()))
images, labels = sample_batch
label_shapes = {f: tuple(t.shape) for f, t in labels.items()}
print(f"Image tensor shape : {tuple(images.shape)}")
print(f"Label tensor shapes: {label_shapes}")

n_show = min(8, images.shape[0])
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, ax in enumerate(axes.flat[:n_show]):
    ax.imshow(denorm(images[i]))
    lines = [f"{f}: {INVERSE_LABEL_MAPS[f][labels[f][i].item()]}" for f in FEATURE_NAMES]
    ax.set_title("\\n".join(lines), fontsize=7)
    ax.axis("off")
plt.suptitle("Sample Augmented Training Batch", fontsize=13)
plt.tight_layout()
plt.show()\
"""

s5_md = """\
## Section 5: Model Instantiation

`MultiHeadResNet` wraps a pretrained ResNet18 backbone with four independent \
`Linear(512, 3)` heads, one per Set card feature. The backbone is frozen for \
`freeze_epochs` epochs so only the heads are trained first; the entire network \
is then fine-tuned end-to-end.

This two-phase schedule (freeze → unfreeze) prevents the pretrained features \
from being destroyed by large early gradients before the heads have converged.\
"""

s5_model = """\
model = MultiHeadResNet(freeze_epochs=5, lr=3e-4, weight_decay=1e-2)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = total - trainable

print(f"Total parameters  : {total:,}")
print(f"Trainable (heads) : {trainable:,}  <- backbone frozen for first 5 epochs")
print(f"Frozen (backbone) : {frozen:,}")
print()
print("Classification heads (each: Linear(512 -> 3)):")
for name in FEATURE_NAMES:
    print(f"  {name}")\
"""

s6_md = """\
## Section 6: Training

**Callbacks & logger:**
- `ModelCheckpoint` — saves the best model by `val_pma` (Perfect Match Accuracy: all 4 features simultaneously correct)
- `EarlyStopping` — halts training if `val_pma` does not improve for 10 consecutive epochs
- `CSVLogger` — writes per-step and per-epoch metrics to a CSV for plotting

**Optimizer & scheduler:** AdamW (`lr=3e-4`, `weight_decay=1e-2`) with \
OneCycleLR that steps once per batch — warm-up, peak, and cosine decay over \
the full training run.

> **Runtime estimate:** Colab T4 ~10 min · MacBook M4 ~25 min · Colab A100 ~3 min \
(30 epochs, online mode with 81 seed images × 300 multiplier).\
"""

s6_setup = """\
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

checkpoint_cb = ModelCheckpoint(
    monitor="val_pma",
    mode="max",
    save_top_k=1,
    dirpath="checkpoints/",
    filename="best-{epoch:02d}-{val_pma:.4f}",
    verbose=True,
)

early_stop_cb = EarlyStopping(monitor="val_pma", mode="max", patience=10, verbose=True)
logger = CSVLogger("logs/", name="set_card_classifier")

trainer = pl.Trainer(
    max_epochs=30,
    accelerator="auto",
    callbacks=[checkpoint_cb, early_stop_cb],
    logger=logger,
    log_every_n_steps=5,
)

print(f"Trainer ready. Logs -> {logger.log_dir}")\
"""

s6_fit = """\
trainer.fit(model, dm)

print()
print(f"Training complete.")
print(f"Best checkpoint : {checkpoint_cb.best_model_path}")
print(f"Best val_pma    : {float(checkpoint_cb.best_model_score):.4f}")\
"""

s6_curves = """\
import pandas as pd

metrics_csv = os.path.join(logger.log_dir, "metrics.csv")
df = pd.read_csv(metrics_csv)

train_loss = df["train_loss_epoch"].dropna().reset_index(drop=True)
val_loss   = df["val_loss"].dropna().reset_index(drop=True)
val_pma    = df["val_pma"].dropna().reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_loss, label="Train Loss")
axes[0].plot(val_loss, label="Val Loss")
axes[0].set_title("Combined Loss (sum of 4 CrossEntropy terms)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(val_pma, color="green", label="Val PMA")
axes[1].set_title("Perfect Match Accuracy (all 4 features correct)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.suptitle("Training Curves", fontsize=14)
plt.tight_layout()
plt.show()\
"""

s7_md = """\
## Section 7: Evaluation

Load the best checkpoint and run the full validation loop to obtain final metrics.

- **`val_f1_{feature}`** — macro-averaged F1 score per head (1.0 = perfect classification)
- **`val_pma`** — Perfect Match Accuracy: fraction of cards where **all four** \
features were simultaneously correct — the primary success metric\
"""

s7_validate = """\
best_model = MultiHeadResNet.load_from_checkpoint(checkpoint_cb.best_model_path)
best_model.eval()

results = trainer.validate(best_model, dm, verbose=False)
metrics = results[0]

print("Validation metrics:")
for k, v in sorted(metrics.items()):
    print(f"  {k:<25s}: {v:.4f}")\
"""

s7_table = """\
import pandas as pd  # may already be imported; harmless to re-import

rows = [
    {"Feature": f.capitalize(), "F1 Score": f"{metrics[f'val_f1_{f}']:.4f}"}
    for f in FEATURE_NAMES
]
rows.append({"Feature": "ALL (Perfect Match)", "F1 Score": f"{metrics['val_pma']:.4f}"})

print(pd.DataFrame(rows).to_string(index=False))\
"""

s8_md = """\
## Section 8: Inference Demo

`predict()` accepts a model and an image path (or a NumPy array) and returns \
a dict with one entry per feature, each containing `prediction` \
(human-readable label string) and `confidence` (softmax probability, 0–1).\
"""

s8_infer = """\
sample_paths = random.sample(glob.glob(os.path.join(RAW_DIR, "*.jpg")), min(6, 81))
print(f"Running inference on {len(sample_paths)} sample cards:\\n")

for path in sample_paths:
    fname = os.path.basename(path)
    parts = fname.split("_")
    true_labels = {
        "color":   parts[0],
        "shape":   parts[1],
        "number":  parts[2],
        "shading": parts[3].split(".")[0],
    }
    result = predict(best_model, path)
    print(f"Image : {fname}")
    for feature, info in result.items():
        match = "OK" if info["prediction"] == true_labels[feature] else "!!"
        print(f"  {feature:8s}: {info['prediction']:10s} ({info['confidence']:.1%}) {match}")
    print()\
"""

s9_md = """\
## Section 9: Visual Debugger

`visualize_predictions` renders a grid of card images annotated per-feature:
- **Green** — prediction matches ground truth
- **Red** — prediction is wrong; shows `ground_truth -> prediction` \
so errors are easy to diagnose at a glance\
"""

s9_viz = """\
# Pull one validation batch
val_images, val_labels = next(iter(dm.val_dataloader()))

# Run the model on the whole batch
best_model = best_model.to(DEVICE)
best_model.eval()
val_images_device = val_images.to(DEVICE)

with torch.no_grad():
    logits = best_model(val_images_device)

B = val_images.shape[0]

# Build one prediction dict per image
predictions = []
for i in range(B):
    pred = {}
    for feature in FEATURE_NAMES:
        probs = torch.softmax(logits[feature][i], dim=0)
        conf, idx = probs.max(dim=0)
        pred[feature] = {
            "prediction": INVERSE_LABEL_MAPS[feature][idx.item()],
            "confidence": round(conf.item(), 4),
        }
    predictions.append(pred)

# Convert integer labels back to human-readable strings
ground_truths = [
    {f: INVERSE_LABEL_MAPS[f][val_labels[f][i].item()] for f in FEATURE_NAMES}
    for i in range(B)
]

# The visualizer accepts CxHxW float32 arrays and de-normalises them internally
images_np = [val_images[i].numpy() for i in range(B)]

fig = visualize_predictions(images_np, ground_truths, predictions, max_images=16)
plt.show()\
"""

s10_md = """\
## Section 10: Model Export

Export the trained model using `torch.export` — the modern PyTorch 2.x \
graph-capture API. The resulting `.pt2` file:
- Captures the full computation graph (backbone + all four heads)
- Excludes all Lightning scaffolding (trainer, metrics, callbacks)
- Can be loaded and run for inference **without PyTorch Lightning installed**
- Is suitable for deployment in a server, container, or edge device\
"""

s10_export = """\
export_path = "checkpoints/model.pt2"

# Export — move model to CPU first for maximum compatibility
ep = export_model(best_model.cpu(), output_path=export_path)
print(f"Model exported to: {export_path}")

# Verify: reload and confirm output shapes
loaded_ep = torch.export.load(export_path)
inference_model = loaded_ep.module()
inference_model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    out = inference_model(dummy_input)

print("\\nExported model output shapes:")
for feature, tensor in out.items():
    print(f"  {feature:8s}: {tuple(tensor.shape)}")

print("\\nExport verified. Ready for deployment without PyTorch Lightning.")\
"""

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------

cells = [
    md("md-00", title),
    md("md-01", s1_md),
    code("code-01", s1_install),
    code("code-02", s1_path),
    code("code-03", s1_imports),
    md("md-02", s2_md),
    code("code-04", s2_bootstrap),
    md("md-03", s3_md),
    code("code-05", s3_dist),
    code("code-06", s3_samples),
    md("md-04", s4_md),
    code("code-07", s4_dm),
    code("code-08", s4_batch),
    md("md-05", s5_md),
    code("code-09", s5_model),
    md("md-06", s6_md),
    code("code-10", s6_setup),
    code("code-11", s6_fit),
    code("code-12", s6_curves),
    md("md-07", s7_md),
    code("code-13", s7_validate),
    code("code-14", s7_table),
    md("md-08", s8_md),
    code("code-15", s8_infer),
    md("md-09", s9_md),
    code("code-16", s9_viz),
    md("md-10", s10_md),
    code("code-17", s10_export),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
        "colab": {"provenance": []},
    },
    "cells": cells,
}

os.makedirs("notebooks", exist_ok=True)
out_path = "notebooks/set_card_classifier_main.ipynb"
with open(out_path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Written: {out_path}  ({len(cells)} cells)")
