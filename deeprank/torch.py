import torch
import torch.nn as nn

__all__ = ["OrdinalOutput"]


class OrdinalOutput(nn.Module):
    """Ordinal regression output layer.

    Projects an arbitrary input down to a single logit and converts it
    into *output_dim* class probabilities via learned, sorted thresholds.

    The layer learns ``output_dim - 1`` interior thresholds ``t(1)…t(K-1)``
    (with ``t(0) = -∞`` and ``t(K) = +∞`` fixed) and computes::

        P(y = k | x) = σ(t(k+1) - logit) - σ(t(k) - logit)

    Args:
        input_dim: Size of the input feature dimension.
        output_dim: Number of ordinal classes.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, 1)
        self.interior_thresholds = nn.Parameter(torch.empty(output_dim - 1))
        self._init_thresholds()

    def _init_thresholds(self):
        nn.init.xavier_uniform_(self.interior_thresholds.unsqueeze(0))
        with torch.no_grad():
            self.interior_thresholds.copy_(self.interior_thresholds.sort().values)

    def forward(self, x):
        logit = self.linear(x)  # (batch, 1)
        t_low = torch.full((1,), float("-inf"), device=logit.device, dtype=logit.dtype)
        t_high = torch.full((1,), float("inf"), device=logit.device, dtype=logit.dtype)
        thresholds = torch.cat([t_low, self.interior_thresholds, t_high])  # (K+1,)
        return torch.sigmoid(thresholds[1:] - logit) - torch.sigmoid(thresholds[:-1] - logit)
