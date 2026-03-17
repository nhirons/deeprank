import torch
import torch.nn as nn

__all__ = ["OrdinalOutput", "ordinal_loss", "ordistic_loss"]


def _penalty(z, name):
    if name == "hinge":
        return torch.clamp(1 - z, min=0)
    elif name == "smooth_hinge":
        return torch.where(
            z >= 1,
            torch.zeros_like(z),
            torch.where(z > 0, (1 - z) ** 2 / 2, 0.5 - z),
        )
    elif name == "modified_least_squares":
        return torch.where(z >= 1, torch.zeros_like(z), (1 - z) ** 2)
    elif name == "logistic":
        return torch.nn.functional.softplus(-z)
    else:
        raise ValueError(f"Unknown penalty: {name}")


def ordinal_loss(logits, targets, thresholds, construction="all", penalty="logistic"):
    """Rennie & Srebro ordinal loss (IJCAI 2005).

    Args:
        logits: (batch,) or (batch, 1) — raw predictor output z(x).
        targets: (batch,) — integer labels in [0, K).
        thresholds: (K-1,) — sorted interior thresholds.
        construction: ``'all'`` or ``'immediate'``.
        penalty: ``'hinge'``, ``'smooth_hinge'``, ``'modified_least_squares'``,
            or ``'logistic'``.

    Returns:
        Scalar mean loss over the batch.
    """
    logits = logits.reshape(-1)
    targets = targets.long()
    K = thresholds.shape[0] + 1
    # Paper uses 1-indexed labels; convert 0-indexed targets
    y = targets + 1  # (batch,) in [1, K]

    if construction == "all":
        # eq 13: sum over l=1..K-1 of f(s(l;y) * (theta_l - z))
        # s(l;y) = -1 if l < y, +1 if l >= y
        l_idx = torch.arange(1, K, device=logits.device).float()  # (K-1,)
        signs = torch.where(l_idx.unsqueeze(0) < y.unsqueeze(1), -1.0, 1.0)  # (batch, K-1)
        diff = thresholds.unsqueeze(0) - logits.unsqueeze(1)  # (batch, K-1)
        loss = _penalty(signs * diff, penalty).sum(dim=1)  # (batch,)
    elif construction == "immediate":
        # eq 12: f(z - theta_{y-1}) + f(theta_y - z)
        t_low = torch.cat([torch.tensor([float("-inf")], device=thresholds.device), thresholds])
        t_high = torch.cat([thresholds, torch.tensor([float("inf")], device=thresholds.device)])
        theta_low = t_low[targets]   # theta_{y-1} (0-indexed: targets maps to y-1)
        theta_high = t_high[targets]  # theta_y
        loss = _penalty(logits - theta_low, penalty) + _penalty(theta_high - logits, penalty)
    else:
        raise ValueError(f"Unknown construction: {construction}")

    return loss.mean()


def ordistic_loss(logits, targets, means, log_priors=None):
    """Ordistic loss (Rennie & Srebro, Section 4).

    Args:
        logits: (batch,) or (batch, 1) — raw predictor output z(x).
        targets: (batch,) — integer labels in [0, K).
        means: (K,) — class means (mu_1=-1, mu_K=1 by convention; interior learned).
        log_priors: (K,) or None — log-prior terms pi_i. Defaults to zeros.

    Returns:
        Scalar mean negative log-likelihood over the batch.
    """
    logits = logits.reshape(-1)
    targets = targets.long()
    K = means.shape[0]
    if log_priors is None:
        log_priors = torch.zeros(K, device=logits.device, dtype=logits.dtype)
    # energy_ik = mu_k * z_i + pi_k - mu_k^2 / 2
    energy = means.unsqueeze(0) * logits.unsqueeze(1) + log_priors.unsqueeze(0) - means.unsqueeze(0) ** 2 / 2
    # loss = -log P(y|z) = -energy[y] + log(sum_k exp(energy[k]))
    target_energy = energy[torch.arange(len(targets), device=targets.device), targets]
    log_partition = torch.logsumexp(energy, dim=1)
    return (log_partition - target_energy).mean()


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
