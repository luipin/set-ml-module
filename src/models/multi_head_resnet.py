import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models


# The four features of a Set card, in the same order used by LABEL_MAPS
# in set_card_data_pipeline.py. Defined at module level so other modules
# (metrics, inference, tests) can import this single source of truth.
FEATURE_NAMES = ('color', 'shape', 'number', 'shading')

# Every feature has exactly 3 possible values (red/green/purple, etc.)
NUM_CLASSES = 3


class MultiHeadResNet(pl.LightningModule):
    """Multi-task CNN that classifies all four Set card features simultaneously.

    Uses a shared ResNet18 backbone (pre-trained on ImageNet) with four
    independent linear classification heads — one per card feature. This
    Multi-Task Learning (MTL) design lets all four tasks share the cost
    of feature extraction while specializing their final predictions.

    Architecture::

        Input (B, 3, 224, 224)
            |
        ResNet18 backbone (conv layers + Global Average Pooling)
            |
        Feature vector (B, 512)
            |
        ┌───────┬────────┬────────┬─────────┐
        │ color │ shape  │ number │ shading │   <- 4 parallel Linear(512, 3) heads
        └───────┴────────┴────────┴─────────┘
            |       |        |         |
        logits (B, 3) per head

    Args:
        freeze_epochs: Number of epochs to keep the backbone frozen.
            During these epochs only the four heads receive gradient updates,
            which prevents large early gradients from corrupting the carefully
            pre-trained backbone weights. After freeze_epochs the full network
            is fine-tuned end-to-end with a much lower learning rate.
            Set to 0 to disable freezing entirely.
    """

    def __init__(self, freeze_epochs: int = 5):
        super().__init__()

        # Saves freeze_epochs to self.hparams and to the checkpoint file,
        # so the model can be reconstructed from a checkpoint without
        # needing to remember which hyperparameters were used.
        self.save_hyperparameters()

        self.freeze_epochs = freeze_epochs

        # --- Backbone ---
        # ResNet18 pre-trained on ImageNet (1.28M images, 1000 classes).
        # The weights=IMAGENET1K_V1 argument requires torchvision >= 0.13.
        # These weights contain rich low-level features (edges, textures,
        # colour gradients) that transfer well to our card images, which
        # also have clear structure and distinct colours.
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Replace ResNet18's final layer (Linear(512, 1000), for ImageNet
        # classification) with a no-op Identity. After this substitution,
        # backbone(x) returns the 512-dimensional feature vector produced
        # by the built-in Global Average Pooling layer — which is exactly
        # what our four heads need as input.
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # --- Four classification heads ---
        # Each head is a single Linear(512, 3): one per Set card feature.
        # nn.ModuleDict registers them properly as submodules, ensuring their
        # parameters are included in model.parameters() and appear in
        # model.state_dict() for checkpointing.
        self.heads = nn.ModuleDict({
            name: nn.Linear(512, NUM_CLASSES)
            for name in FEATURE_NAMES
        })

        # Freeze the backbone at initialisation so the first freeze_epochs
        # epochs only train the four heads. This is a standard transfer
        # learning warm-up strategy.
        if self.freeze_epochs > 0:
            self._freeze_backbone()

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self):
        """Freezes all backbone parameters (no gradient updates).

        Called at __init__ when freeze_epochs > 0. Frozen parameters are
        excluded from the optimiser's update step, so only the four heads
        learn during the warm-up phase.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self):
        """Unfreezes all backbone parameters for end-to-end fine-tuning.

        Called automatically by on_train_epoch_start once freeze_epochs
        have elapsed. After this point the full network trains together,
        allowing the backbone to adapt its features to Set card images.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def on_train_epoch_start(self):
        """Unfreezes the backbone once freeze_epochs epochs have completed.

        Lightning calls this hook at the start of every training epoch.
        self.current_epoch is 0-indexed, so freeze_epochs=5 means the
        backbone is frozen for epochs 0–4 and unfrozen from epoch 5 onward.
        """
        if self.freeze_epochs > 0 and self.current_epoch == self.freeze_epochs:
            self._unfreeze_backbone()
            self.print(
                f'Epoch {self.current_epoch}: backbone unfrozen for end-to-end fine-tuning.'
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Runs the shared backbone then each classification head.

        Args:
            x: Image batch of shape (B, 3, 224, 224), float32,
                ImageNet-normalized (mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]).

        Returns:
            Dict mapping each feature name to a logit tensor of shape (B, 3).
            Logits are raw pre-softmax scores. Apply softmax to obtain
            per-class probabilities; argmax gives the predicted class index.

        Example::

            model = MultiHeadResNet()
            batch = torch.randn(32, 3, 224, 224)
            logits = model(batch)
            # logits['color'].shape  -> (32, 3)
            # logits['shape'].shape  -> (32, 3)
            # logits['number'].shape -> (32, 3)
            # logits['shading'].shape-> (32, 3)
        """
        # Shared feature extraction: (B, 3, 224, 224) -> (B, 512)
        # The backbone's built-in avgpool reduces the final feature maps
        # from (B, 512, 7, 7) to (B, 512, 1, 1), which flatten gives (B, 512).
        features = self.backbone(x)

        # Each head produces independent logits from the same feature vector.
        # Using a dict return (not a tuple) makes downstream code — loss
        # computation, metric logging, inference — self-documenting.
        return {name: head(features) for name, head in self.heads.items()}
