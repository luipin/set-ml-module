# Offline vs Online Augmentation — Decision Guide

This document explains the two ways to use the augmentation pipeline during training, when each makes sense, and how to configure the code for each approach.

---

## 1. What are offline and online augmentation?

The augmentation pipeline (`get_spatial_color_transforms`) can run in two places: **before training** (offline) or **during training** (online). Both ultimately feed the same augmented images to the model — the difference is *when* the augmentation work happens.

```mermaid
flowchart TD
    RAW["data/raw/\n81 seed images"]

    RAW --> OFF["OFFLINE\nRun bootstrap_dataset.py once\nbefore training starts"]
    RAW --> ON["ONLINE\nAugment live inside\nSetCardDataset.__getitem__"]

    OFF --> DISK["~24,300 JPEG files\nsaved to data/augmented/"]
    DISK --> DM_OFF["SetCardDataModule\ndata_dir='data/augmented'\ndynamic_multiplier=1"]

    ON --> DM_ON["SetCardDataModule\ndata_dir='data/raw'\ndynamic_multiplier=300"]

    DM_OFF --> TRAIN["Model training\n(Lightning Trainer)"]
    DM_ON --> TRAIN
```

In both cases the model sees the same volume of augmented data — approximately 24,300 training images per epoch (81 seeds × 300 variants × 0.8 train split). Only the source and timing differ.

---

## 2. How the code differs

### Offline setup

```bash
# Step 1: run once to generate the augmented dataset
python -m src.data.bootstrap_dataset \
    --raw_dir data/raw \
    --augmented_dir data/augmented \
    --augmentations 300

# Step 2: point DataModule at the pre-generated files
```

```python
dm = SetCardDataModule(
    data_dir='data/augmented',
    batch_size=32,
    num_workers=4,
    dynamic_multiplier=1,   # files already expanded on disk
)
```

### Online setup

```python
# No bootstrap step needed — just point at the 81 seed images
dm = SetCardDataModule(
    data_dir='data/raw',
    batch_size=32,
    num_workers=4,
    dynamic_multiplier=300, # repeat each seed 300x per epoch
)
```

The `dynamic_multiplier` works by list multiplication in `SetCardDataModule.setup()`:

```python
train_paths = train_paths * self.dynamic_multiplier
# 65 train seeds × 300 = 19,500 items in the epoch
```

Each repeated path produces a *different* augmented image because `get_train_transforms()` randomizes every call.

---

## 3. Where each approach spends its compute

```mermaid
flowchart LR
    subgraph OFFLINE["Offline"]
        direction TB
        B1["bootstrap_dataset.py\nAll augmentation CPU work\nhappens here — once"] --> D1["JPEG files on disk"]
        D1 --> T1["Training loop\nReads files → Resize\n→ Normalize → ToTensorV2\n(lightweight per batch)"]
    end

    subgraph ONLINE["Online"]
        direction TB
        T2["Training loop\nReads 81 files → Full augmentation\n→ Resize → Normalize → ToTensorV2\n(heavier per batch, every epoch)"]
    end
```

**Offline**: heavy CPU work is paid once upfront. Each training step does minimal work (disk read + normalize + tensor).

**Online**: heavy CPU work is paid every single batch of every epoch. The same augmentation that bootstrap ran 300× total per seed is now run 300× per seed *per epoch restart* — except in practice each epoch is a new random pass, which is exactly what you want.

---

## 4. Is disk IO a bottleneck?

In general, disk IO is one of the most common training bottlenecks in ML — but **not for this project**, for two reasons:

### Dataset size vs available RAM

| Pipeline | Files | Total size | Fits in 16 GB RAM? |
|---|---|---|---|
| Online | 81 files | ~3 MB | Yes — cached after the first epoch in seconds |
| Offline | ~24,300 files | ~700 MB | Yes — fully cached within 1–2 epochs |

Once the OS loads files into its page cache (RAM), subsequent reads bypass the disk entirely. For this dataset, the cache fills on the very first epoch and stays warm for the rest of training.

### SSD read speed

An M4 MacBook Air has an NVMe SSD capable of ~3–5 GB/s sequential reads. Even the offline 700 MB dataset would take less than 1 second to read cold. Disk IO would only become a real constraint at hundreds of GBs (e.g., ImageNet, video datasets).

---

## 5. Speed comparison

For this project, the dominant cost is the **GPU forward and backward pass**, not augmentation or disk IO. The difference between offline and online is small in practice.

### What actually drives training time

```mermaid
flowchart LR
    BATCH["Fetch batch\nfrom DataLoader"] --> AUG["Augment\n(CPU, parallel\nworkers)"]
    AUG --> XFER["Transfer to GPU\n(PCIe or unified memory)"]
    XFER --> FWD["Forward pass\nResNet18 + 4 heads\n(GPU)"]
    FWD --> LOSS["Compute loss\n(GPU)"]
    LOSS --> BWD["Backward pass\ngradient computation\n(GPU)"]
    BWD --> OPT["Optimizer step\n(GPU)"]

    style FWD fill:#d4edda
    style BWD fill:#d4edda
    style OPT fill:#d4edda
```

The green steps (forward + backward + optimizer) dominate. With `num_workers >= 4`, the CPU augmentation work runs in parallel and keeps the GPU continuously fed — the GPU never has to wait for augmentation to finish.

### Rough training time estimates

For 20 epochs on ~19,000 training images, batch size 32:

| Hardware | Throughput | Per epoch | 20 epochs |
|---|---|---|---|
| MacBook Air M4 (MPS) | ~150–300 img/s | 1–2 min | **20–40 min** |
| Colab T4 (free tier) | ~800–1,500 img/s | 15–45 sec | **5–15 min** |
| Colab A100 (Pro) | ~3,000–5,000 img/s | 5–15 sec | **2–5 min** |

These estimates apply to both offline and online. The difference between them is a rounding error compared to the hardware gap.

---

## 6. The M4 unified memory advantage

The MacBook Air M4 has a hardware characteristic that makes online augmentation unusually efficient: **unified memory** — the CPU and GPU share the same physical memory pool.

```mermaid
flowchart TB
    subgraph DISCRETE["Discrete GPU system (Colab T4/A100)"]
        direction LR
        CM["CPU RAM\n(system memory)"] -->|"PCIe bus\n~16–32 GB/s"| GM["GPU VRAM\n(dedicated)"]
    end

    subgraph UNIFIED["Apple M4 (unified memory)"]
        direction LR
        UM["Shared memory pool\n(CPU + GPU both access)"]
        CPU["CPU cores"] --- UM
        GPU["GPU cores"] --- UM
    end
```

On discrete GPU systems, augmented tensors travel from CPU RAM to GPU VRAM over the PCIe bus — a real transfer cost. On M4, the augmented tensor created by the CPU is already in the same memory space the GPU reads from. There is no transfer. This partially offsets the cost of running augmentations live on the CPU.

---

## 7. Decision guide

```mermaid
flowchart TD
    START["Which pipeline should I use?"]

    START --> Q1{"Is disk space\na concern?"}
    Q1 -->|"No (have 1+ GB free)"| Q2{"Do you want the\nsimplest setup?"}
    Q1 -->|"Yes (tight on disk)"| ONLINE["Use ONLINE\nOnly 81 files needed"]

    Q2 -->|Yes| ONLINE
    Q2 -->|No — want to pre-compute once| OFFLINE["Use OFFLINE\nRun bootstrap once,\nfaster per-step afterward"]

    OFFLINE --> Q3{"Did profiling show\naugmentation is a\nbottleneck?"}
    Q3 -->|"No (GPU is still the bottleneck)"| NOTE["Both approaches perform\nequivalently — stick with online\nfor simplicity next time"]
    Q3 -->|Yes| KEEP["Offline is the right call"]
```

### Summary table

| | Offline | Online |
|---|---|---|
| Run `bootstrap_dataset.py`? | Yes, once | No |
| `data_dir` | `data/augmented` | `data/raw` |
| `dynamic_multiplier` | `1` | `300` |
| Extra disk usage | ~700 MB | None |
| Setup steps | 2 (bootstrap + train) | 1 (train) |
| Speed difference | Marginal | Marginal |
| Variety per epoch | Fixed — same 24,300 files every epoch | True random — new augmentation per epoch |
| Best for | Very slow augmentation or huge datasets | Everything else |

### Recommendation for this project

**Use online.** The dataset is tiny, the hardware can keep up, and online augmentation produces strictly more varied training data (a different random augmentation every epoch rather than the same 300 fixed variants repeated forever). Only switch to offline if profiling shows the CPU is starving the GPU.

---

## 8. One subtle advantage of online: true randomness per epoch

There is one qualitative reason to prefer online beyond convenience. With offline augmentation, each seed image maps to exactly 300 fixed variants — and the model will see the same 300 images every single epoch. With online, the model sees a freshly randomized variant every time it encounters a seed. Over 20 epochs, it effectively sees 20× more variety.

```mermaid
flowchart LR
    subgraph OFFLINE_VAR["Offline: fixed variants"]
        S1["Seed image"] --> V1["Variant 001 (fixed)"]
        S1 --> V2["Variant 002 (fixed)"]
        S1 --> V3["Variant 003 (fixed)"]
        V1 -->|"Epoch 2"| V1
        V2 -->|"Epoch 2"| V2
        V3 -->|"Epoch 2"| V3
    end

    subgraph ONLINE_VAR["Online: new random each epoch"]
        S2["Seed image"] -->|"Epoch 1"| R1["Random variant A"]
        S2 -->|"Epoch 2"| R2["Random variant B\n(different from A)"]
        S2 -->|"Epoch 3"| R3["Random variant C\n(different again)"]
    end
```

For a dataset this small (81 seeds), this extra variety meaningfully reduces the chance of the model memorizing specific augmented images rather than learning general features.
