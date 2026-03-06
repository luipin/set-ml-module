# Set Card Classifier (CNN)

This project implements a Deep Learning system designed to classify images of cards from the popular game "Set". The model uses a Multi-Task Learning (MTL) approach to predict the four distinct features of a Set card simultaneously from a single image.

## Project Overview

In the game of Set, every card possesses four independent features, each with three possible values:
* **Color:** Red, Green, Purple
* **Shape:** Diamond, Squiggle, Oval
* **Number:** 1, 2, 3
* **Shading:** Solid, Striped, Open

Instead of training four separate neural networks or a massive 81-class classifier, this project uses a **Multi-Task CNN Architecture**. It leverages a shared ResNet18 backbone to extract fundamental visual features (edges, colors, textures) and passes them to four parallel classification "heads" to predict the individual properties.

### Key Technologies
* **Language:** Python 3.10+
* **Framework:** PyTorch & PyTorch Lightning
* **Augmentation:** Albumentations
* **Architecture:** ResNet18 (Pre-trained) with custom Multi-Head Top

---

## Folder Structure

```text
set-ml-module/
├── checkpoints/                        # Trained model artifacts (git-ignored)
│   ├── best-epoch=XX-val_pma=Y.ckpt   # PyTorch Lightning checkpoint (training artifact)
│   └── model.pt                        # TorchScript export (deployment artifact)
├── data/                               # Datasets (git-ignored)
│   ├── raw/                            # 81 seed images — {color}_{shape}_{number}_{shading}.jpg
│   └── augmented/                      # Bootstrapped dataset (16k–40k images)
├── doc/                                # Architecture and design notes
│   ├── augmentation-pipeline.md
│   ├── multi-head-resnet-training.md
│   └── offline-vs-online-augmentation.md
├── logs/                               # CSVLogger training metrics (git-ignored)
├── notebooks/                          # Jupyter notebooks
│   ├── set_card_classifier_main.ipynb  # End-to-end: data → train → evaluate → export
│   ├── inference.ipynb                 # Inference with the .ckpt checkpoint
│   ├── pt2_deployment.ipynb            # Deployment demo using the TorchScript .pt export
│   ├── 01-data-exploration.ipynb       # Dataset exploration and augmentation visualisation
│   └── 02-background-augmentation-test.ipynb
├── scripts/
│   └── generate_notebook.py            # Utility to scaffold notebook cells
├── src/                                # Source package
│   ├── data/
│   │   ├── set_card_data_pipeline.py   # SetCardDataset + SetCardDataModule (LightningDataModule)
│   │   └── bootstrap_dataset.py        # Offline augmentation script (generates data/augmented/)
│   ├── models/
│   │   ├── multi_head_resnet.py        # MultiHeadResNet (LightningModule) — backbone + 4 heads
│   │   ├── predictor.py                # predict() and predict_batch() inference helpers
│   │   └── export.py                   # export_model() — TorchScript export via torch.jit.trace
│   └── utils/
│       ├── augmentations.py            # Albumentations train/val transform pipelines
│       ├── metrics.py                  # PerfectMatchAccuracy + per-feature F1
│       └── visualizer.py              # visualize_predictions() grid display
├── tests/                              # Unit tests
│   ├── test_data_pipeline.py
│   ├── test_model_architecture.py
│   ├── test_predictor.py
│   ├── test_export.py
│   └── test_visualizer.py
├── .venv/                              # Virtual environment (git-ignored)
├── requirements.txt                    # Python dependencies
└── set-card-classifier-spec.md         # Original project specification
```

> **Checkpoint formats:** `checkpoints/best-*.ckpt` is the training checkpoint — it stores weights, optimizer state, and hyperparameters and can resume training. `checkpoints/model.pt` is the TorchScript deployment artifact — it has no Lightning dependency, is ~43 MB, and accepts any batch size.

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/luipin/set-ml-module.git
   cd set-ml-module
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Provide the Seed Images:**
   To train the model, you need to provide the base 81 images representing every unique Set card.
   * Place the 81 images into the `data/raw/` directory.
   * **Crucial Naming Convention:** Each image *must* be named according to its features.
     * Format: `{color}_{shape}_{number}_{shading}.jpg`
     * Example: `red_diamond_1_solid.jpg`

4. **Bootstrap the Dataset (Optional):**
   You can pre-generate the large training dataset offline using the bootstrap script:
   ```bash
   python -m src.data.bootstrap_dataset --raw_dir data/raw --augmented_dir data/augmented --augmentations 300
   ```
   *Note: The `SetCardDataModule` also supports dynamic (on-the-fly) augmentation if you prefer not to save thousands of files to disk.*
