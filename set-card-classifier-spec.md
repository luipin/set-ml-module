# **Specification: Set Card Feature Classifier (CNN)**

## **1\. Project Overview**

Create a Deep Learning system to classify images of cards from the game "Set".

Each card possesses 4 independent features, each with 3 possible values:

* **Color:** Red, Green, Purple  
* **Shape:** Diamond, Squiggle, Oval  
* **Number:** 1, 2, 3  
* **Shading:** Solid, Striped, Open

The system utilizes a **Multi-Task Learning (MTL)** approach: a single CNN backbone (ResNet18) with four specialized "heads" to predict each feature simultaneously.

## **2\. Technical Stack**

* **Language:** Python 3.10+  
* **Frameworks:** PyTorch & PyTorch Lightning  
* **Augmentation:** Albumentations  
* **Environment:** Google Colab (.ipynb)  
* **Architecture:** ResNet18 (Pre-trained) with custom Multi-Head Top.

## **3\. Engineering & Documentation Standards**

* **Google-Style Docstrings:** Required for every class and function.  
* **Educational Comments:** Explain hyperparameters (learning rates, weight decay) and augmentation choices.  
* **Robust Error Handling:** \- Verify 81 seed images exist before starting.  
  * Validate filename format: {color}\_{shape}\_{number}\_{shading}.jpg.  
  * Handle missing or corrupted files during the data pipeline.

## **4\. Milestone 0: Data Bootstrapping & Strategy**

* **Seed Dataset:** Start with a folder of 81 unique images (one per card combination).  
* **Naming Convention:** red\_diamond\_1\_solid.jpg, purple\_squiggle\_3\_open.jpg, etc.  
* **Augmentation Target:** Generate **200–500 variations per card**.  
* **Total Training Set:** Approximately 16,000 to 40,000 images.  
* **Organization:** Flattened structure; labels are parsed directly from filenames.

## **5\. Milestone 1: Data Augmentation Pipeline**

Implement a LightningDataModule utilizing albumentations.

* **Logic:** Simulate real-world variance from single-source images.  
* **Key Transforms:**  
  * ShiftScaleRotate: Handles misalignment.  
  * RandomBrightnessContrast: Handles lighting variance.  
  * HueSaturationValue: **Strict limits** (+/- 15\) to maintain color label integrity.  
  * Normalization: Use ImageNet constants (mean=\[0.485, 0.456, 0.406\]).

## **6\. Milestone 2: Multi-Task CNN Architecture**

Implement the model as a LightningModule.

* **Backbone:** ResNet18 (frozen for initial epochs, then fine-tuned).  
* **Multi-Head Design:** \- Shared Backbone \-\> Global Average Pooling \-\> 4 Parallel Linear Layers.  
  * Each Linear head outputs a vector of size 3 (logits).  
* **Forward Pass:** Returns a dictionary: {'color': ..., 'shape': ..., 'number': ..., 'shading': ...}.

## **7\. Milestone 3: Training & Evaluation**

* **Combined Loss:** ![][image1].  
* **Optimizer:** AdamW with OneCycleLR scheduler for fast convergence in Colab.  
* **Metrics:** \- F1-Score per feature.  
  * **Perfect Match Accuracy:** Counts as 1.0 only if all 4 features match.

## **8\. Milestone 4: Inference, Visualization & Export**

* **Predictor Function:** Returns human-readable JSON with confidence scores.  
* **Visual Debugger:** A grid display highlighting errors in **red** labels.  
* **Model Export:** Code to export the final model to **TorchScript** for deployment.

## **9\. Deliverables**

* Fully documented Google Colab Notebook.  
* set\_card\_data\_pipeline.py (Completed).  
* Final model weights (.ckpt or .pth).

[image1]: <data:image/png;base64,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>