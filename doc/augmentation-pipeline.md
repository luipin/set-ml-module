# Augmentation Pipeline — Deep Dive

This document explains how image augmentation works in this project, from first principles up to the full pipeline. It is written for someone comfortable with Python who has not used Albumentations before.

---

## 1. What is Albumentations and why do we use it?

Albumentations is a library for **image augmentation** — the practice of generating new training images by applying random transformations to existing ones. The goal is to make the model robust to real-world variance: different lighting, camera angles, zoom levels, and backgrounds.

The key things to know before reading any code:

- Every augmentation is a **Transform object** (e.g. `A.ShiftScaleRotate`, `A.GaussianBlur`)
- Transforms are chained together using **`A.Compose([...])`**, which runs them in order
- Every transform has a **`p=` parameter** (probability), which controls whether it fires on any given image. `p=1.0` means always apply; `p=0.7` means apply 70% of the time, skip 30%
- You call the composed pipeline like a function, passing the image as a keyword argument:
  ```python
  result = transform(image=my_numpy_array)
  augmented_image = result['image']
  ```
- The image is always a **NumPy array** with shape `(H, W, 3)`, dtype `uint8`, pixel values 0–255
- The output is a **dict** — `result['image']` gives you the augmented array

### The `p=` parameter — visualized

```mermaid
flowchart LR
    A["Input image\nnumpy array (H×W×3)"] --> B["A.Compose runs\neach transform in order"]
    B --> C["Transform 1\np=0.7"]
    C -->|"70%: dice roll succeeds"| D["Transform applied ✓"]
    C -->|"30%: dice roll fails"| E["Image passed through unchanged"]
    D --> F["Transform 2\np=0.3"]
    E --> F
    F -->|"30%: applied"| G["..."]
    F -->|"70%: skipped"| G
    G --> H["Final output dict\n{ 'image': result }"]
```

Each transform makes its own independent dice roll. In a single forward pass through the pipeline, some transforms will fire and some won't — this is what makes each augmented image unique.

---

## 2. The two augmentation contexts

This project uses augmentation in **two different places** with different goals:

```mermaid
flowchart TD
    RAW["data/raw/\n81 seed images (.jpg)"]

    RAW --> BOOT["bootstrap_dataset.py\nOffline — runs once"]
    RAW --> TRAIN["SetCardDataset.__getitem__\nOnline — runs every epoch"]

    BOOT --> SC["get_spatial_color_transforms()\nSpatial + color only\nNo tensor conversion"]
    SC --> SAVE["Saves ~24,300 .jpg files\nto data/augmented/"]

    TRAIN --> WHICH{Which split?}
    WHICH -->|"Training set"| TT["get_train_transforms()\nSpatial + color\n+ Normalize\n+ ToTensorV2"]
    WHICH -->|"Validation set"| VT["get_val_transforms()\nNormalize only\n+ ToTensorV2"]

    TT --> TENSOR1["torch.Tensor (3×H×W)\nfloat32, ImageNet-normalized"]
    VT --> TENSOR2["torch.Tensor (3×H×W)\nfloat32, ImageNet-normalized"]

    TENSOR1 --> MODEL["ResNet18 MultiHead Model"]
    TENSOR2 --> MODEL

    SAVE --> TRAIN
```

**Offline (bootstrap):** Run once before training. Takes each of the 81 seed images and generates ~300 variations, saving them to disk as `.jpg` files. Uses only spatial and color transforms — no tensor conversion, because the result is saved as an image file, not fed to a model.

**Online (training):** Runs every time `__getitem__` is called (i.e., every time the DataLoader fetches a sample). The loaded image is augmented on-the-fly, normalized, and converted to a PyTorch tensor. Validation images skip augmentation entirely — only normalization and tensor conversion are applied.

---

## 3. The three transform pipelines

There are three functions in `augmentations.py`. Here is exactly what each one does and why.

### 3.1 `get_spatial_color_transforms()`

```python
A.Compose([
    A.ShiftScaleRotate(..., p=0.7),
    AddRandomBackground(..., p=1.0),
    A.RandomBrightnessContrast(..., p=0.7),
    A.HueSaturationValue(..., p=0.7),
    A.GaussianBlur(..., p=0.3),
])
```

This is the **core augmentation recipe** — the transforms that simulate real-world variance. It produces a numpy array as output (no tensor conversion).

```mermaid
flowchart TD
    IN["Input: numpy array (H×W×3)\nA real-world Set card image"]

    IN --> SSR["ShiftScaleRotate\np=0.7\n\nRandomly shifts, zooms, or rotates\nthe card within the frame.\nEmpty areas filled with solid black (value=0)."]

    SSR --> ARB["AddRandomBackground\np=1.0 — always runs\n\nReplaces black/dark pixels with\na random background texture."]

    ARB --> RBC["RandomBrightnessContrast\np=0.7\n\nSimulates different lighting conditions:\nbright room, dim room, direct sunlight."]

    RBC --> HSV["HueSaturationValue\np=0.7\n\nSlightly shifts color tone, saturation, vividness.\nHUE IS CAPPED AT ±15 to prevent\nred from drifting to purple or green."]

    HSV --> GB["GaussianBlur\np=0.3\n\nAdds a mild blur to simulate\ncamera shake or out-of-focus shots."]

    GB --> OUT["Output: numpy array (H×W×3)\nAugmented image, still 0–255 uint8"]
```

**Why this exact order?** Order matters. `ShiftScaleRotate` must run first because it introduces black padding around the card. `AddRandomBackground` must run second to replace that black padding with a texture — if it ran first, there would be nothing to replace yet.

### 3.2 `get_train_transforms()`

```python
A.Compose([
    get_spatial_color_transforms(),   # the recipe above, nested inside
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

This wraps the spatial/color pipeline and adds two final steps required by PyTorch:

```mermaid
flowchart TD
    IN["Input: numpy (H×W×3)\nuint8, values 0–255"]
    IN --> SC["get_spatial_color_transforms()\nAll 5 transforms above"]
    SC --> NORM["A.Normalize\n\nConverts each pixel from 0–255 range\nto a float centered around ImageNet stats.\n\nPixel = (pixel/255 - mean) / std\n\nResult: float32, roughly in range -2.0 to +2.5"]
    NORM --> T2["ToTensorV2\n\nReshapes from (H, W, C) to (C, H, W)\nand converts to torch.Tensor.\n\nPyTorch expects channels-first format."]
    T2 --> OUT["Output: torch.Tensor (3×H×W)\nfloat32, ready for ResNet18"]
```

**Why normalize?** ResNet18 was pre-trained on ImageNet. The weights in its early layers were calibrated expecting inputs normalized with specific mean and standard deviation values. If we feed it raw 0–255 integers, the activations will be on the wrong scale and the pre-trained weights won't transfer well.

**Why `ToTensorV2`?** NumPy arrays are `(H, W, C)` — height first, channels last. PyTorch expects `(C, H, W)` — channels first. `ToTensorV2` handles this reshape. It also avoids an unnecessary copy (unlike the old `ToTensor`).

### 3.3 `get_val_transforms()`

```python
A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

**No spatial or color augmentation.** The validation set must represent the model's performance on natural images — images the model hasn't seen before. If we augmented validation images, we'd be evaluating on a different distribution than what we trained on, and the metrics would be misleading.

```mermaid
flowchart LR
    TR["Training pipeline"]
    VA["Validation pipeline"]

    TR --> T1["ShiftScaleRotate ✓"]
    TR --> T2["AddRandomBackground ✓"]
    TR --> T3["RandomBrightnessContrast ✓"]
    TR --> T4["HueSaturationValue ✓"]
    TR --> T5["GaussianBlur ✓"]
    TR --> T6["Normalize ✓"]
    TR --> T7["ToTensorV2 ✓"]

    VA --> V1["ShiftScaleRotate ✗"]
    VA --> V2["AddRandomBackground ✗"]
    VA --> V3["RandomBrightnessContrast ✗"]
    VA --> V4["HueSaturationValue ✗"]
    VA --> V5["GaussianBlur ✗"]
    VA --> V6["Normalize ✓"]
    VA --> V7["ToTensorV2 ✓"]
```

---

## 4. The `AddRandomBackground` custom transform — in detail

This is the only custom transform in the project. Albumentations lets you write your own by subclassing `A.ImageOnlyTransform` and implementing the `apply()` method.

The problem it solves: raw seed images have dark rounded corners (the card shape doesn't fill the whole rectangle), and after `ShiftScaleRotate` rotates the card, more dark/black area is introduced around the edges. A model trained on these would learn that "black borders = a Set card" — not a generalizable rule. This transform replaces those dark areas with varied backgrounds.

```mermaid
flowchart TD
    IN["Input: numpy array (H×W×3)\nCard image with possible dark corners\nor black padding from rotation"]

    IN --> GRAY["Step 1: Convert to grayscale\ncv2.cvtColor RGB→GRAY\n\nWhy grayscale? We only need brightness\nto find dark pixels, not color."]

    GRAY --> THRESH["Step 2: Threshold at value 30\ncv2.threshold THRESH_BINARY_INV\n\nPixels with brightness < 30 → white (255) in mask\nPixels with brightness ≥ 30 → black (0) in mask\n\nThis mask marks every dark/background pixel."]

    THRESH --> BGDEC{"Step 3: Decide what\nbackground to generate"}

    BGDEC -->|"bg_dir provided\nAND random() < 0.7\n→ 70% chance"| REAL["Load a real background image\nfrom the bg_dir folder\n\nPick one at random, resize\nit to match the card dimensions."]

    BGDEC -->|"No bg_dir OR\nrandom() ≥ 0.7\n→ 30% chance"| SYNTH{"Synthetic background"}

    SYNTH -->|"random() < 0.5\n→ 50% of 30%"| SOLID["Solid random color\nnp.random.randint(50, 255, 3)\n\nOne color fills the entire\nbackground uniformly."]

    SYNTH -->|"random() ≥ 0.5\n→ 50% of 30%"| NOISE["Random noise\nnp.random.randint(50, 255, H×W×3)\n\nEvery pixel is a different\nrandom color — looks like static."]

    REAL --> BLEND
    SOLID --> BLEND
    NOISE --> BLEND

    BLEND["Step 4: Blend into masked areas\nnp.where(mask, background, original)\n\nWhere mask=True (dark pixel): use background value\nWhere mask=False (card pixel): keep original value"]

    BLEND --> OUT["Output: numpy array (H×W×3)\nCard pixels unchanged,\ndark corners replaced with background"]
```

### The `np.where` blend — explained

`np.where(condition, x, y)` is like a per-pixel if/else:
- Where `condition` is `True` → use value from `x` (the background)
- Where `condition` is `False` → use value from `y` (the original image)

The mask is `True` at every dark pixel, so those get the background. The card itself (brighter pixels) keeps its original values. No pixels are "mixed" — it's a hard swap.

### The 70%/50% probability tree

```mermaid
flowchart TD
    START["AddRandomBackground fires\np=1.0 in this project"]
    START --> Q1{"bg_dir provided\nAND roll < 0.7?"}
    Q1 -->|"Yes (70% if bg_dir exists)"| REAL["Real background image\nfrom disk"]
    Q1 -->|"No (100% if no bg_dir,\n30% if bg_dir exists)"| Q2{"roll < 0.5?"}
    Q2 -->|"Yes (50%)"| SOLID["Solid random color"]
    Q2 -->|"No (50%)"| NOISE["Random pixel noise"]
```

In our project, `bg_dir` is `None` in all current usages (no real background images are provided), so the real-image branch never fires. Every call picks either a solid color or noise with equal probability.

---

## 5. The HueSaturationValue constraint — why it matters

This is the most important design decision in the augmentation pipeline.

Set cards have three possible colors: **red**, **green**, and **purple**. These are the labels the model must predict. If augmentation shifts a red card's hue too far, it might start looking purple — corrupting the label without changing it.

Hue is measured on a 0–360° wheel:

```
        Yellow (~60°)
           |
Green (~120°) ——— Red (0°/360°)
           |
        Cyan (~180°)
           |
Blue (~240°) ——— Magenta (~300°)
           |
        Purple (~270°)
```

Red and purple are only ~90° apart. A `hue_shift_limit=15` means we allow at most a ±15° shift. This keeps red firmly in the red zone and purple firmly in the purple zone with a large safety margin.

If we used `hue_shift_limit=90` (a common default), a red card could become orange, yellow, or even greenish — the label "red" would be wrong.

```mermaid
flowchart LR
    RED["Red card\nhue ≈ 0°"] --> SHIFT["HueSaturationValue\nhue_shift_limit = 15\n\nShifts hue by at most ±15°"]
    SHIFT --> SAFE["Result: hue between -15° and +15°\nStill clearly red ✓"]

    RED2["Red card\nhue ≈ 0°"] --> SHIFT2["Hypothetical unsafe transform\nhue_shift_limit = 90"]
    SHIFT2 --> UNSAFE["Result: hue could reach 90°\nLooks yellow or green — label is wrong ✗"]
```

---

## 6. End-to-end data flow during training

This ties everything together — what happens from raw image file to model input:

```mermaid
flowchart TD
    FILE["data/augmented/red_diamond_1_solid_aug_0042.jpg\nStored on disk"]

    FILE --> CV2["cv2.imread()\nLoads as BGR numpy array"]
    CV2 --> RGB["cv2.cvtColor BGR→RGB\nAlbumentations expects RGB"]
    RGB --> PARSE["Parse filename\nred → 0, diamond → 0, 1 → 0, solid → 0\nLabels become torch.long tensors"]
    RGB --> AUG["get_train_transforms()(image=array)\n\n1. ShiftScaleRotate (p=0.7)\n2. AddRandomBackground (p=1.0)\n3. RandomBrightnessContrast (p=0.7)\n4. HueSaturationValue (p=0.7)\n5. GaussianBlur (p=0.3)\n6. Normalize (always)\n7. ToTensorV2 (always)"]

    AUG --> TENSOR["torch.Tensor (3×H×W)\nfloat32, ImageNet-normalized"]
    PARSE --> LABELS["labels dict\n{ color: tensor(0),\n  shape: tensor(0),\n  number: tensor(0),\n  shading: tensor(0) }"]

    TENSOR --> BATCH["DataLoader collects 32 samples into a batch"]
    LABELS --> BATCH

    BATCH --> MODEL["ResNet18 MultiHead Model\nInput: (32, 3, H, W)"]
    MODEL --> OUT["Output dict\n{ color: (32,3), shape: (32,3),\n  number: (32,3), shading: (32,3) }"]
```

---

## 7. Known gap: missing Resize step

Currently neither `get_train_transforms()` nor `get_val_transforms()` includes a `Resize(224, 224)` step. ResNet18 expects 224×224 input. If the raw images are a different size, PyTorch will either crash or silently produce wrong results (depending on whether the sizes happen to work with the convolution strides).

This is tracked as **Issue #1**. The fix is to add `A.Resize(224, 224)` as the first step in both pipelines.
