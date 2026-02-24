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

The project is structured to separate data pipelines, model architecture, and interactive exploration (Jupyter Notebooks).

```text
set-ml-module/
├── checkpoints/                # Saved model weights (TorchScript models)
├── data/                       # Dataset directories (ignored by git)
│   ├── raw/                    # The 81 original seed images (e.g., red_diamond_1_solid.jpg)
│   └── augmented/              # The bootstrapped dataset (16,000 - 40,000 images)
├── notebooks/                  # Interactive experimentation
│   ├── 01-data-exploration.ipynb      # For visualizing augmentations
│   └── set_card_classifier_main.ipynb # Full training and evaluation notebook (Deliverable)
├── src/                        # Source code package
│   ├── data/
│   │   ├── bootstrap_dataset.py       # Script to generate augmented images offline
│   │   └── set_card_data_pipeline.py  # PyTorch LightningDataModule and Dataset
│   ├── models/
│   │   └── multi_head_resnet.py       # The PyTorch LightningModule (Multi-Task CNN)
│   └── utils/
│       ├── augmentations.py           # Albumentations transform pipelines
│       └── metrics.py                 # Custom metrics (F1, Perfect Match Accuracy)
├── tests/                      # Unit tests for the pipeline and model
│   ├── test_data_pipeline.py
│   └── test_model_architecture.py
├── requirements.txt            # Python dependencies
└── set-card-classifier-spec.md # The original project specification document
```

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/luipin/set-ml-module.git
   cd set-ml-module
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment (e.g., `venv` or `conda`).
   ```bash
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
