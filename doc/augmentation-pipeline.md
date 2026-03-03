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

This section walks through the complete chain — from calling `Trainer.fit()` all the way to a batch arriving at the model — explaining every layer of the machinery.

---

### 6.1 The four-layer stack

There are four components working together. Understanding who calls whom is essential before reading any of the code.

```mermaid
flowchart TD
    LT["PyTorch Lightning Trainer\nTrainer.fit(model, datamodule)\n\nOrchestrates the training loop.\nCalls datamodule.setup(), then\nasks for dataloaders each epoch."]

    LT --> DM["SetCardDataModule\nset_card_data_pipeline.py:97\n\nResponsible for:\n- Finding image files on disk\n- Splitting into train/val\n- Creating Dataset objects\n- Returning DataLoaders"]

    DM --> DL["PyTorch DataLoader\n(one for train, one for val)\n\nResponsible for:\n- Shuffling (train only)\n- Fetching samples in parallel\n- Grouping samples into batches\n- Calling __getitem__ repeatedly"]

    DL --> DS["SetCardDataset\nset_card_data_pipeline.py:19\n\nResponsible for:\n- Loading one image from disk\n- Parsing its label from filename\n- Running the augmentation transform\n- Returning (tensor, labels_dict)"]
```

The DataLoader is the engine that drives the Dataset. It calls `__getitem__` on the Dataset repeatedly, collects the results, and stacks them into batches. The Trainer drives the DataLoader — it asks for the next batch, passes it to the model, computes the loss, and updates the weights.

---

### 6.2 `SetCardDataModule.setup()` — what runs before training starts

`setup()` is called once by the Trainer before the first epoch. It does three things:

```mermaid
flowchart TD
    SETUP["DataModule.setup()\nset_card_data_pipeline.py:120"]

    SETUP --> GLOB["Step 1: File discovery\nglob('data/augmented/*.jpg')\nglob('data/augmented/*.png')\n\nBuilds a flat list of every\nimage path in the directory.\nRaises FileNotFoundError if empty."]

    GLOB --> SPLIT["Step 2: 80/20 train/val split\ntrain_test_split(all_images, test_size=0.2, random_state=42)\n\nScikit-learn shuffles the list\nand cuts it: 80% train, 20% val.\nrandom_state=42 makes the split\nreproducible — same split every run."]

    SPLIT --> MULT["Step 3: Dynamic multiplier\ntrain_paths = train_paths * dynamic_multiplier\n\nPython list multiplication: repeats the\nlist N times. If multiplier=300 and\nyou have 65 train images:\n65 × 300 = 19,500 paths in the list.\nSame 65 files, each listed 300 times."]

    MULT --> DS1["SetCardDataset(train_paths, transform=get_train_transforms())\nDataset for training — with augmentation"]
    SPLIT --> DS2["SetCardDataset(val_paths, transform=get_val_transforms())\nDataset for validation — normalize only"]
```

**The dynamic multiplier explained:**

The multiplier is a trick for when you're training on the 81 raw seed images directly (not the bootstrapped augmented set). Without it, one epoch would be only 65 training steps (81 × 0.8 = 65 images). That's too short — the model barely sees anything before the epoch ends and validation runs.

With `dynamic_multiplier=300`, the path list is `[img1, img2, ..., img65, img1, img2, ...]` — 300 repetitions. Each call to `__getitem__` loads the same file but runs a fresh random augmentation, producing a different image each time. So it's 19,500 iterations per epoch, each one a unique augmented view of the 65 training cards.

When using the bootstrapped `data/augmented/` folder (~24,000 files), the multiplier should be left at `1` — there are already enough unique files.

---

### 6.3 The DataLoader — how batching works

The DataLoader sits between the Dataset and the model. It controls how samples are fetched and assembled.

```mermaid
flowchart TD
    DL["DataLoader\nbatch_size=32, shuffle=True, num_workers=4"]

    DL --> SHUF["shuffle=True (training only)\n\nAt the start of each epoch, the DataLoader\nrandomizes the order of all image paths.\nThis prevents the model from memorizing\nthe sequence of cards."]

    DL --> WORK["num_workers=4\n\nSpawns 4 parallel worker processes.\nEach worker calls __getitem__ independently.\nWhile the GPU is processing batch N,\nworkers are already loading batch N+1.\nWithout this, data loading would starve the GPU."]

    DL --> FETCH["For each batch:\nCalls __getitem__(idx) 32 times\nfor 32 different indices"]

    FETCH --> COLL["Collation (automatic)\n\nDataLoader stacks the 32 individual\nsamples into tensors:\n\n32 × (3,H,W) tensors → one (32,3,H,W) tensor\n32 × {color:scalar} dicts → {color:(32,)} tensor\n\nThis happens automatically via default_collate."]

    COLL --> BATCH["One batch:\n  images: torch.Tensor (32, 3, H, W)\n  labels: {\n    'color':   torch.Tensor (32,)  ← 32 class indices\n    'shape':   torch.Tensor (32,)\n    'number':  torch.Tensor (32,)\n    'shading': torch.Tensor (32,)\n  }"]
```

**Why `shuffle=False` for validation?** Validation order doesn't affect the results (we're just measuring accuracy, not updating weights), and a fixed order makes runs reproducible and easier to debug.

---

### 6.4 `SetCardDataset.__getitem__` — what happens for a single sample

This is the method the DataLoader calls 32 times to build one batch. Here is every line, explained:

```mermaid
flowchart TD
    IDX["DataLoader requests index 42\n__getitem__(42)"]

    IDX --> PATH["img_path = self.image_paths[42]\n→ 'data/augmented/red_diamond_1_solid_aug_0042.jpg'"]

    PATH --> READ["cv2.imread(img_path)\n\nOpens the .jpg from disk.\nReturns a numpy array (H, W, 3), dtype=uint8.\n\nCRITICAL: OpenCV loads images in BGR order\n(Blue, Green, Red) — not the RGB order\nthat Albumentations and humans expect."]

    READ --> CHECK["if image is None: raise ValueError\n\nOpenCV returns None (not an exception)\nif the file is missing or corrupted.\nWe check explicitly and raise a clear error."]

    CHECK --> BGR2RGB["cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n\nSwaps channel order: BGR → RGB.\nNow pixel [r, g, b] has the correct values.\nAlbumentations and torchvision both expect RGB."]

    BGR2RGB --> PARSE["Filename parsing\nos.path.basename → 'red_diamond_1_solid_aug_0042.jpg'\n.split('_') → ['red', 'diamond', '1', 'solid', 'aug', '0042.jpg']\n\nparts[0] = 'red'\nparts[1] = 'diamond'\nparts[2] = '1'\nparts[3] = 'solid'  ← .split('.')[0] drops the extension\n                       (or the aug suffix if it were here)"]

    PARSE --> MAP["LABEL_MAPS lookup\n\n'red'     → LABEL_MAPS['color']['red']     = 0\n'diamond' → LABEL_MAPS['shape']['diamond'] = 0\n'1'       → LABEL_MAPS['number']['1']      = 0\n'solid'   → LABEL_MAPS['shading']['solid'] = 0\n\nEach integer wrapped in torch.tensor(..., dtype=torch.long)\ntorch.long = 64-bit integer, required by CrossEntropyLoss"]

    MAP --> LABELS["labels = {\n  'color':   tensor(0),\n  'shape':   tensor(0),\n  'number':  tensor(0),\n  'shading': tensor(0)\n}"]

    BGR2RGB --> AUG["self.transform(image=rgb_array)\n\nPasses the numpy (H,W,3) array to\nthe Albumentations pipeline.\nReturns dict: {'image': result}"]

    AUG --> TENSOR["augmented['image']\n→ torch.Tensor shape (3, H, W)\n   dtype float32\n   values roughly -2.0 to +2.5\n   (after normalization)"]

    TENSOR --> RETURN["return image_tensor, labels\n\nA tuple of:\n  - torch.Tensor (3, H, W)\n  - dict of 4 scalar tensors"]
    LABELS --> RETURN
```

---

### 6.5 LABEL_MAPS — why integers, and why these specific values

```python
LABEL_MAPS = {
    'color':   {'red': 0, 'green': 1, 'purple': 2},
    'shape':   {'diamond': 0, 'squiggle': 1, 'oval': 2},
    'number':  {'1': 0, '2': 1, '3': 2},
    'shading': {'solid': 0, 'striped': 1, 'open': 2}
}
```

**Why integers?** PyTorch's `CrossEntropyLoss` (and `F1Score` metrics) require integer class indices, not strings. The model outputs a vector of 3 logits per head — index 0 means "class 0", index 1 means "class 1", etc. The label must be one of those indices so the loss can compare them.

**Why these specific integers?** The mapping is arbitrary — what matters is consistency. `red=0` just means "red is class zero in the color head." As long as every sample with a red card maps to `0`, and the model learns to output its highest logit at index `0` for red cards, the classifier is correct.

**Why `dtype=torch.long`?** `CrossEntropyLoss` requires the target to be a 64-bit integer tensor (`torch.long`). If you pass a float or a 32-bit int, PyTorch will raise a runtime error. This is enforced at label creation time so the error surfaces early.

---

### 6.6 What a complete batch looks like — shapes at every stage

```mermaid
flowchart LR
    S1["Single sample from __getitem__\n\nimage:  Tensor (3, H, W)\nlabels: {\n  color:   Tensor scalar\n  shape:   Tensor scalar\n  number:  Tensor scalar\n  shading: Tensor scalar\n}"]

    S1 -->|"×32 samples\nDataLoader stacks"| BATCH["One batch (batch_size=32)\n\nimages:  Tensor (32, 3, H, W)\nlabels: {\n  color:   Tensor (32,)\n  shape:   Tensor (32,)\n  number:  Tensor (32,)\n  shading: Tensor (32,)\n}"]

    BATCH -->|"→ ResNet18 backbone"| FEAT["Feature map\n(32, 512, 1, 1)\nafter Global Avg Pool"]

    FEAT -->|"→ 4 linear heads"| OUT["Model output dict\n{\n  color:   Tensor (32, 3)  ← 3 logits per sample\n  shape:   Tensor (32, 3)\n  number:  Tensor (32, 3)\n  shading: Tensor (32, 3)\n}"]
```

The `(32, 3)` output means: for each of the 32 images in the batch, the model produced 3 raw scores (logits). The highest score is the predicted class. `CrossEntropyLoss` compares these logits against the integer label (e.g., `0` for red) to compute the loss.

---

### 6.7 Full picture — one training step from disk to loss

```mermaid
flowchart TD
    DISK["data/augmented/\n~24,300 .jpg files on disk"]

    DISK --> SETUP["DataModule.setup()\nDiscover files → 80/20 split → apply multiplier\nCreate train Dataset + val Dataset"]

    SETUP --> EPOCH["Start of epoch\nDataLoader shuffles training indices"]

    EPOCH --> WORKERS["4 parallel worker processes\nEach worker calls __getitem__ independently"]

    WORKERS --> GETITEM["__getitem__(idx) — per sample:\n1. cv2.imread → BGR numpy\n2. BGR → RGB\n3. filename.split('_') → label integers\n4. transform(image=array) → Tensor (3,H,W)\n5. return (tensor, labels_dict)"]

    GETITEM --> COLLATE["DataLoader collation\nStack 32 samples:\nimages → (32,3,H,W)\nlabels → 4× (32,) tensors"]

    COLLATE --> FORWARD["model.training_step(batch)\nForward pass through ResNet18\nOutput: 4 × (32,3) logit tensors"]

    FORWARD --> LOSS["Combined loss\nloss = CrossEntropy(color_logits, color_labels)\n     + CrossEntropy(shape_logits, shape_labels)\n     + CrossEntropy(number_logits, number_labels)\n     + CrossEntropy(shading_logits, shading_labels)"]

    LOSS --> BACK["loss.backward()\nAdamW.step()\nWeights updated"]

    BACK --> NEXT["Next batch →\nrepeat until epoch ends"]

    NEXT --> VAL["Validation\nSame flow, but:\n- No shuffle\n- get_val_transforms() only\n- No weight updates"]
```

---

## 7. Known gap: missing Resize step

Currently neither `get_train_transforms()` nor `get_val_transforms()` includes a `Resize(224, 224)` step. ResNet18 expects 224×224 input. If the raw images are a different size, PyTorch will either crash or silently produce wrong results (depending on whether the sizes happen to work with the convolution strides).

This is tracked as **Issue #1**. The fix is to add `A.Resize(224, 224)` as the first step in both pipelines.
