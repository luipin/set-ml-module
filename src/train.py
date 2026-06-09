"""Command-Line Interface (CLI) script to train the Multi-Task Set Card Classifier.

Optimized for Apple Silicon (M4/MPS) hardware backends. Performs training,
logs metrics, evaluates the best checkpoint, and exports to TorchScript.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# Ensure project root is in the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.set_card_data_pipeline import SetCardDataModule
from src.models.multi_head_resnet import MultiHeadResNet, FEATURE_NAMES
from src.models.export import export_model


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the Multi-Task Set Card Classifier model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths & parameters
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to dataset directory. If omitted, auto-detects augmented or raw directories."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Number of images per batch."
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of CPU worker threads for data loading. If omitted, sets automatically."
    )
    parser.add_argument(
        "--dynamic-multiplier", type=int, default=None,
        help="Repeat train set N times per epoch. Auto-detected (300 for raw, 1 for augmented)."
    )

    # Model hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Maximum training epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4,
        help="Peak learning rate for AdamW."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-2,
        help="L2 weight decay regularization."
    )
    parser.add_argument(
        "--freeze-epochs", type=int, default=5,
        help="Number of initial epochs to keep the ResNet backbone frozen."
    )
    
    # Runtime options
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience (epochs with no validation PMA improvement)."
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory where CSV logs are saved."
    )
    parser.add_argument(
        "--checkpoints-dir", type=str, default="checkpoints",
        help="Directory where model checkpoints are saved."
    )
    
    return parser.parse_args()


def auto_detect_dataset(args):
    """Dynamically resolves directories and repeats count depending on files availability."""
    augmented_path = PROJECT_ROOT / "data/augmented"
    raw_path = PROJECT_ROOT / "data/raw"
    
    if args.data_dir:
        data_path = Path(args.data_dir).resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory '{args.data_dir}' does not exist.")
        # If user passed raw explicitly, default multiplier is 300, otherwise 1
        default_mult = 300 if "raw" in data_path.name.lower() else 1
        multiplier = args.dynamic_multiplier if args.dynamic_multiplier is not None else default_mult
        data_mode = "manual"
    elif augmented_path.exists() and len(list(augmented_path.glob("*.jpg"))) > 81:
        data_path = augmented_path
        multiplier = args.dynamic_multiplier if args.dynamic_multiplier is not None else 1
        data_mode = "offline-augmented"
    elif raw_path.exists() and len(list(raw_path.glob("*.jpg"))) == 81:
        data_path = raw_path
        multiplier = args.dynamic_multiplier if args.dynamic_multiplier is not None else 300
        data_mode = "online-augmented (on-the-fly)"
    else:
        raise FileNotFoundError(
            "Could not auto-detect dataset. Ensure you have either:\n"
            "  1. Exactly 81 seed images in 'data/raw/' (naming format: red_diamond_1_solid.jpg)\n"
            "  2. Bootstrapped/augmented images in 'data/augmented/'\n"
            "  Or pass --data-dir explicitly."
        )
        
    return data_path, multiplier, data_mode


def main():
    args = parse_arguments()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    # 1. Resolve dataset configurations
    try:
        data_dir, multiplier, data_mode = auto_detect_dataset(args)
    except FileNotFoundError as e:
        print(f"\nConfiguration Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Resolve CPU workers. On macOS, default to 0 to prevent torch_shm_manager
    # shared memory socket errors (common due to sandboxing and OS-level limitations).
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if sys.platform == "darwin" else min(8, os.cpu_count() or 4)
        
    # Print hardware and run information
    print("==================================================")
    print("     SET CARD CLASSIFIER CLI TRAINING SUITE       ")
    print("==================================================")
    print(f"Data Mode:      {data_mode}")
    print(f"Data Folder:    {data_dir}")
    print(f"Multiplier:     {multiplier}x per epoch")
    print(f"Batch Size:     {args.batch_size}")
    print(f"CPU Workers:    {num_workers} threads")
    
    # Check for Apple Silicon GPU acceleration
    if torch.backends.mps.is_available():
        accelerator = "mps"
        device_name = "Apple Silicon GPU (MPS)"
    elif torch.cuda.is_available():
        accelerator = "gpu"
        device_name = "Nvidia CUDA GPU"
    else:
        accelerator = "cpu"
        device_name = "Standard CPU (No hardware acceleration)"
        
    print(f"Hardware Dev:   {device_name}")
    print(f"Learning Rate:  {args.lr}")
    print(f"Freeze Epochs:  {args.freeze_epochs}")
    print("--------------------------------------------------")

    # Set up directories
    checkpoints_path = PROJECT_ROOT / args.checkpoints_dir
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Instantiate Data Pipeline
    logging.info("Initializing Data Module...")
    dm = SetCardDataModule(
        data_dir=str(data_dir),
        batch_size=args.batch_size,
        num_workers=num_workers,
        dynamic_multiplier=multiplier
    )
    
    # 3. Instantiate Multi-Task Model
    logging.info("Instantiating MultiHeadResNet Model...")
    model = MultiHeadResNet(
        freeze_epochs=args.freeze_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 4. Configure Callbacks and Loggers
    # ModelCheckpoint saves the weights with the highest Perfect Match Accuracy (PMA)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_path),
        filename="best-{epoch:02d}-{val_pma:.4f}",
        monitor="val_pma",
        mode="max",
        save_top_k=1,
        save_last=True
    )
    
    # EarlyStopping stops training if PMA doesn't improve for `patience` epochs
    early_stop_callback = EarlyStopping(
        monitor="val_pma",
        patience=args.patience,
        mode="max"
    )
    
    # CSV Logger writes clean training rows into metrics.csv
    logger = CSVLogger(save_dir=args.log_dir, name="set_card_classifier")
    
    # 5. Instantiate Trainer and Start Training Loop
    logging.info("Starting Training Loop...")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        enable_progress_bar=True
    )
    
    start_time = time.time()
    trainer.fit(model, dm)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    logging.info("Training complete.")

    # 6. Retrieve best model checkpoint and export to TorchScript
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        # Fallback if training exited immediately or didn't save checkpoint
        best_model_path = str(checkpoints_path / "last.ckpt")
        
    logging.info(f"Loading best checkpoint from '{best_model_path}' for export...")
    try:
        best_model = MultiHeadResNet.load_from_checkpoint(best_model_path)
    except Exception as e:
        logging.error(f"Failed to load best checkpoint: {e}. Falling back to last model state.")
        best_model = model

    export_path = checkpoints_path / "model.pt"
    logging.info(f"Tracing and exporting best model to TorchScript: '{export_path}'")
    try:
        export_model(best_model, output_path=str(export_path))
    except Exception as e:
        logging.error(f"Failed to export TorchScript model: {e}")

    # 7. Print Performance Summary
    # Load logged metrics from CSV to display validation history
    metrics_csv = Path(logger.log_dir) / "metrics.csv"
    best_pma = 0.0
    final_train_loss = 0.0
    epochs_run = 0
    f1_scores = {}

    if metrics_csv.exists():
        try:
            df = pd.read_csv(metrics_csv)
            epochs_run = df["epoch"].max() + 1
            
            # Find best validation PMA
            if "val_pma" in df.columns:
                best_pma = df["val_pma"].max()
                
            # Find last training loss
            if "train_loss_epoch" in df.columns:
                final_train_loss = df["train_loss_epoch"].dropna().iloc[-1]
                
            # Retrieve final validation F1 scores per feature
            for feature in FEATURE_NAMES:
                col = f"val_f1_{feature}"
                if col in df.columns:
                    f1_scores[feature] = df[col].dropna().iloc[-1]
        except Exception as e:
            logging.warning(f"Could not parse training logs from metrics.csv: {e}")

    # Format elapsed time
    mins, secs = divmod(int(elapsed_time), 60)
    
    print("\n==================================================")
    print("              TRAINING RUN SUMMARY                ")
    print("==================================================")
    print(f"Total Epochs Run:   {epochs_run}")
    print(f"Execution Duration: {mins:02d}m {secs:02d}s")
    if epochs_run > 0:
        print(f"Average Epoch Time: {elapsed_time / epochs_run:.2f}s")
    print(f"Final Train Loss:   {final_train_loss:.4f}")
    print(f"Best Val PMA:       {best_pma:.2%} (Perfect Match Accuracy)")
    print("--------------------------------------------------")
    print("Final Feature-level F1 Validation Scores:")
    for feature in FEATURE_NAMES:
        score = f1_scores.get(feature, 0.0)
        print(f"  * {feature.capitalize():<8}: {score:.2%}")
    print("--------------------------------------------------")
    print(f"Saved Checkpoint:   {best_model_path}")
    print(f"Exported Model:     {export_path.resolve()}")
    print("==================================================")


if __name__ == "__main__":
    main()
