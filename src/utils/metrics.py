import torch
import torchmetrics


class PerfectMatchAccuracy(torchmetrics.Metric):
    """Accuracy that requires all four Set card features to be correct simultaneously.

    A prediction scores 1.0 only when color, shape, number, AND shading are
    all correct for that sample. This is stricter than averaging per-feature
    accuracies: a model that gets 3 of 4 features right every time scores 0%
    here, not 75%.

    Example::

        pma = PerfectMatchAccuracy()
        preds  = {'color': tensor([0, 1]), 'shape': tensor([1, 0]),
                  'number': tensor([2, 1]), 'shading': tensor([0, 2])}
        target = {'color': tensor([0, 1]), 'shape': tensor([1, 1]),  # sample 1 shape wrong
                  'number': tensor([2, 1]), 'shading': tensor([0, 2])}
        pma.update(preds, target)
        pma.compute()  # tensor(0.5)  — 1 out of 2 samples fully correct
    """

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total",   default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: dict[str, torch.Tensor], target: dict[str, torch.Tensor]) -> None:
        """Accumulate perfect-match counts for one batch.

        Args:
            preds:  Dict mapping feature name to a (B,) tensor of predicted class indices.
            target: Dict mapping feature name to a (B,) tensor of ground-truth class indices.
                    Must have the same keys and batch size as preds.
        """
        pred_stack   = torch.stack(list(preds.values()),  dim=1)  # (B, num_tasks)
        target_stack = torch.stack(list(target.values()), dim=1)  # (B, num_tasks)
        perfect = (pred_stack == target_stack).all(dim=1)         # (B,) bool
        self.correct += perfect.sum()
        self.total   += perfect.shape[0]

    def compute(self) -> torch.Tensor:
        """Returns the fraction of samples where all features were predicted correctly."""
        return self.correct.float() / self.total
