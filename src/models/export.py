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
    output_path: str = "model.pt2",
    example_input: torch.Tensor | None = None,
) -> torch.export.ExportedProgram:
    """Exports a trained MultiHeadResNet using ``torch.export`` for deployment.

    Uses ``torch.export`` (introduced in PyTorch 2.0), the modern graph-capture
    API that supersedes the older ``torch.jit.trace`` / TorchScript approach.
    The exported ``.pt2`` file can be reloaded without PyTorch Lightning
    installed, making it suitable for standalone inference services.

    The Lightning-specific layers (metrics, trainer hooks) are excluded from
    the export by wrapping only the backbone and heads in a plain ``nn.Module``
    before exporting. The exported model is therefore lighter and more portable.

    Args:
        model: A trained ``MultiHeadResNet`` instance. The model is moved to
            eval mode before export so batch normalisation layers behave
            correctly at inference time.
        output_path: Destination path for the exported ``.pt2`` file. Parent
            directories are created automatically. If a ``.pt`` suffix is
            given it is silently changed to ``.pt2`` (the format required by
            ``torch.export.save``). Defaults to ``"model.pt2"``.
        example_input: A float32 tensor of shape ``(B, 3, 224, 224)`` used to
            trace the computation graph. If ``None``, a single zero tensor
            ``(1, 3, 224, 224)`` is created on the same device as the model.
            Defaults to ``None``.

    Returns:
        torch.export.ExportedProgram: The exported program. It can be used
        immediately for inference or reloaded later with ``torch.export.load()``.

    Raises:
        ValueError: If ``example_input`` is provided but does not have shape
            ``(B, 3, 224, 224)``.

    Example — exporting::

        from src.models.multi_head_resnet import MultiHeadResNet
        from src.models.export import export_model

        model = MultiHeadResNet.load_from_checkpoint("checkpoints/best.ckpt")
        export_model(model, output_path="checkpoints/model.pt2")

    Example — loading and running the exported model (no Lightning required)::

        import torch

        ep = torch.export.load("checkpoints/model.pt")
        inference_model = ep.module()
        inference_model.eval()

        image = torch.randn(1, 3, 224, 224)  # preprocessed card image
        with torch.no_grad():
            logits = inference_model(image)
        # logits is a dict: {"color": Tensor(1,3), "shape": ..., ...}
        color_class = logits["color"].argmax(dim=1).item()
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

    with torch.no_grad():
        ep = torch.export.export(exportable, (example_input,))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # torch.export uses the .pt2 format; rename silently if the caller passed .pt
    if output_path.suffix == ".pt":
        output_path = output_path.with_suffix(".pt2")
    torch.export.save(ep, str(output_path))

    return ep
