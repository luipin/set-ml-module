import torch
import torch.nn as nn
from pathlib import Path

from src.models.multi_head_resnet import MultiHeadResNet, FEATURE_NAMES


class _ExportableModel(nn.Module):
    """Thin nn.Module wrapper containing only the backbone and heads.

    ``torch.export`` cannot handle ``pl.LightningModule`` directly because
    Lightning's ``trainer`` property raises when accessed outside of a training
    context during module inspection. This wrapper exposes the same
    ``forward()`` interface using only plain PyTorch primitives, making it
    safe to export.
    """

    def __init__(self, backbone: nn.Module, heads: nn.ModuleDict):
        super().__init__()
        self.backbone = backbone
        self.heads = heads

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {name: head(features) for name, head in self.heads.items()}


def export_model(
    model: MultiHeadResNet,
    output_path: str = "checkpoints/model.pt",
    example_input: torch.Tensor | None = None,
) -> torch.jit.ScriptModule:
    """Exports a trained MultiHeadResNet as a TorchScript traced model.

    Uses ``torch.jit.trace`` to capture the computation graph (backbone + all
    four classification heads) into a self-contained ``.pt`` file that can be
    loaded and run without PyTorch Lightning installed.

    TorchScript is chosen over ``torch.export`` because it handles dynamic
    batch sizes reliably out of the box. ``torch.export`` bakes the example
    input's batch size as a static shape guard, causing ``AssertionError``
    at inference time whenever the batch size differs — a limitation that
    cannot be worked around consistently across PyTorch versions.

    The Lightning-specific layers (metrics, trainer hooks) are excluded from
    the trace by wrapping only the backbone and heads in a plain ``nn.Module``
    before tracing.

    Args:
        model: A trained ``MultiHeadResNet`` instance. Moved to eval mode
            before tracing so batch normalisation layers use running statistics.
        output_path: Destination path for the ``.pt`` file. Parent directories
            are created automatically. Defaults to ``"checkpoints/model.pt"``.
        example_input: A float32 tensor of shape ``(B, 3, 224, 224)`` used to
            trace the computation graph. If ``None``, a single zero tensor
            ``(1, 3, 224, 224)`` is used. Any batch size works — TorchScript
            traces are batch-size agnostic. Defaults to ``None``.

    Returns:
        torch.jit.ScriptModule: The traced module, usable immediately for
        inference and reloadable with ``torch.jit.load()``.

    Raises:
        ValueError: If ``example_input`` does not have shape ``(B, 3, 224, 224)``.

    Example — exporting::

        from src.models.multi_head_resnet import MultiHeadResNet
        from src.models.export import export_model

        model = MultiHeadResNet.load_from_checkpoint("checkpoints/best.ckpt")
        export_model(model, output_path="checkpoints/model.pt")

    Example — loading and running (no Lightning required)::

        import torch

        model = torch.jit.load("checkpoints/model.pt")
        model.eval()

        image = torch.randn(4, 3, 224, 224)  # batch of 4 preprocessed cards
        with torch.no_grad():
            logits = model(image)
        # logits: {"color": Tensor(4,3), "shape": Tensor(4,3), ...}
        color_classes = logits["color"].argmax(dim=1).tolist()
    """
    if example_input is not None:
        if example_input.ndim != 4 or example_input.shape[1] != 3:
            raise ValueError(
                f"example_input must have shape (B, 3, 224, 224), "
                f"got {tuple(example_input.shape)}."
            )

    model.eval()
    device = next(model.parameters()).device

    if example_input is None:
        example_input = torch.zeros(1, 3, 224, 224, device=device)

    exportable = _ExportableModel(model.backbone, model.heads)
    exportable.eval()

    # torch.jit.trace records ops for the given example but does not bake
    # the batch size as a concrete constant — the traced graph works for
    # any batch size at inference time.
    # strict=False is required because the model returns a dict. The structure
    # is fixed (always the same 4 keys), so this is safe.
    with torch.no_grad():
        traced = torch.jit.trace(exportable, example_input, strict=False)

    # Resolve to absolute path immediately so the save location is unambiguous
    # regardless of the caller's working directory.
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output_path))

    return traced
