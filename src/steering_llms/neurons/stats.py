"""Per-neuron statistics the papers rank neurons by. Pure tensor code, no model.

Inputs are either :class:`~.collect.GroupStats` (token-level moments) or sample matrices
``[n_samples, ...units]`` (one pooled value per text). Every function is vectorized over units,
so ``[n_layers, width]`` or flat ``[units]`` both work.
"""

import math

import torch
from torch import Tensor

from .collect import GroupStats

EPS = 1e-6


# ---- from token-level moments ------------------------------------------------------------------------


def freq_diff(a: GroupStats, b: GroupStats) -> Tensor:
    """Firing-rate difference ``Pr_a(x > 0) - Pr_b(x > 0)`` (NPTI's δ)."""
    return a.frac_pos - b.frac_pos


def count_diff(a: GroupStats, b: GroupStats) -> Tensor:
    """Difference in the number of tokens with ``x > 0`` (sNeuron-TST, before its top-k)."""
    return (a.pos - b.pos).float()


def mean_diff(a: GroupStats, b: GroupStats) -> Tensor:
    return a.mean - b.mean


def cohens_d(a: GroupStats, b: GroupStats) -> Tensor:
    """``(mean_a - mean_b) / sqrt((var_a + var_b) / 2)`` (DPN-LE's d)."""
    return (a.mean - b.mean) / ((a.var + b.var) / 2).sqrt().clamp_min(EPS)


def lape_entropy(groups: list[GroupStats]) -> Tensor:
    """Entropy over groups of the normalized firing rate; low means the neuron fires for few labels.

    Language-activation-probability entropy, used by Emotion & Rhetoric neurons: with
    ``p_k = rate_k / sum_k rate_k``, ``H = -sum_k p_k log p_k``. Neurons that never fire get the
    maximum entropy so they never look specific.
    """
    rates = torch.stack([g.frac_pos for g in groups])  # [K, ...]
    total = rates.sum(0, keepdim=True)
    p = rates / total.clamp_min(EPS)
    h = -(p * p.clamp_min(1e-12).log()).sum(0)
    return torch.where(total[0] > 0, h, torch.full_like(h, math.log(len(groups))))


def zscore(x: Tensor) -> Tensor:
    """Standardize over every unit (the gender composite z-scores across all neurons)."""
    x = x.float()
    return (x - x.mean()) / x.std().clamp_min(EPS)


# ---- from sample matrices ----------------------------------------------------------------------------


def _units(x: Tensor) -> tuple[Tensor, tuple[int, ...]]:
    """``[N, *shape]`` -> (``[units, N]``, shape)."""
    shape = tuple(x.shape[1:])
    return x.reshape(x.shape[0], -1).T, shape


def sample_mean_diff(pos: Tensor, neg: Tensor) -> Tensor:
    return pos.float().mean(0) - neg.float().mean(0)


def sample_cohens_d(pos: Tensor, neg: Tensor) -> Tensor:
    """Cohen's d over samples, pooled with the (unbiased) variance average of the two groups."""
    p, n = pos.float(), neg.float()
    v = (p.var(0) + n.var(0)) / 2
    return (p.mean(0) - n.mean(0)) / v.sqrt().clamp_min(EPS)


@torch.no_grad()
def auroc(pos: Tensor, neg: Tensor, chunk: int = 16384) -> Tensor:
    """Per-unit AUROC of separating ``pos`` [Np, ...] from ``neg`` [Nn, ...], ties counted as 1/2.

    ``P(x_pos > x_neg) + P(x_pos = x_neg) / 2``, computed exactly by sorting ``neg`` and binary
    searching every positive value.
    """
    P, shape = _units(pos.float())
    N, _ = _units(neg.float())
    out = torch.empty(P.shape[0])
    for s in range(0, P.shape[0], chunk):
        n_sorted = N[s : s + chunk].sort(-1).values.contiguous()
        p = P[s : s + chunk].contiguous()
        lt = torch.searchsorted(n_sorted, p, right=False).float()
        le = torch.searchsorted(n_sorted, p, right=True).float()
        out[s : s + chunk] = ((lt + le) / 2).mean(-1).cpu() / N.shape[1]
    return out.reshape(shape)


def gini(au: Tensor) -> Tensor:
    """AurA's Gini coefficient ``2 * (AUROC - 0.5)``; the dampening factor is ``1 - gini``."""
    return 2 * (au - 0.5)


@torch.no_grad()
def average_precision(scores: Tensor, labels: Tensor, chunk: int = 16384) -> Tensor:
    """Per-unit average precision of ``scores`` [N, ...] ranking the ``labels`` [N] (bool) positives first.

    Matches ``sklearn.metrics.average_precision_score`` including ties: tied scores form one
    threshold, and each positive is credited with the precision at the end of its tie group.
    """
    X, shape = _units(scores.float())
    y = labels.bool().to(X.device)
    n_pos = int(y.sum())
    if n_pos == 0:
        return torch.zeros(shape)
    N = X.shape[1]
    ar = torch.arange(N, device=X.device)
    out = torch.empty(X.shape[0])
    for s in range(0, X.shape[0], chunk):
        v, order = X[s : s + chunk].sort(-1, descending=True)
        lab = y[order].float()  # [units, N]
        tp = lab.cumsum(-1)
        prec = tp / (ar + 1)
        end = torch.ones_like(v, dtype=torch.bool)
        end[:, :-1] = v[:, 1:] != v[:, :-1]
        end_idx = torch.where(end, ar.expand_as(v), torch.full_like(order, N))
        end_idx = end_idx.flip(-1).cummin(-1).values.flip(-1)
        prec_end = prec.gather(-1, end_idx)
        out[s : s + chunk] = ((lab * prec_end).sum(-1) / n_pos).cpu()
    return out.reshape(shape)


def quantile_cols(x: Tensor, q: float, dim: int = 0) -> Tensor:
    """``q``-quantile (0..1, linear interpolation) along ``dim``; no size limit, unlike torch.quantile."""
    v = x.float().sort(dim).values
    n = v.shape[dim]
    pos = q * (n - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    a, b = v.select(dim, lo), v.select(dim, hi)
    return a + (b - a) * (pos - lo)


def flat_quantile(x: Tensor, q: float) -> float:
    """``q``-quantile (0..1) of all elements, via kthvalue (works for > 16M elements)."""
    flat = x.reshape(-1).float()
    k = max(1, min(flat.numel(), int(round(q * (flat.numel() - 1))) + 1))
    return flat.kthvalue(k).values.item()


@torch.no_grad()
def l1_logistic(
    X: Tensor,
    y: Tensor,
    lam: float = 1e-3,
    steps: int = 500,
    lr: float | None = None,
    standardize: bool = True,
) -> tuple[Tensor, float]:
    """L1-regularized logistic regression by FISTA; returns (weights [D], bias).

    Minimizes ``mean(log(1 + exp(-s * (X w + b)))) + lam * |w|_1`` with ``s = 2y - 1``. Columns
    are standardized first (weights are for the standardized columns), so large weights mark the
    units that carry the label, as in the sentiment-neuron paper's probe.
    """
    X = X.float()
    s = y.float() * 2 - 1
    if standardize:
        X = (X - X.mean(0)) / X.std(0).clamp_min(EPS)
    n, d = X.shape
    if lr is None:  # 1 / Lipschitz constant of the logistic loss gradient: ||X||_2^2 / (4n)
        sigma = torch.linalg.matrix_norm(X, ord=2).item() if d * n < 5e7 else X.norm().item()
        lr = 4 * n / max(sigma**2, EPS)
    w = torch.zeros(d)
    b = torch.zeros(())
    zw, zb, t = w.clone(), b.clone(), 1.0
    for _ in range(steps):
        m = s * (X @ zw + zb)
        g = -s * torch.sigmoid(-m) / n
        gw, gb = X.T @ g, g.sum()
        w_new = zw - lr * gw
        w_new = w_new.sign() * (w_new.abs() - lr * lam).clamp_min(0)
        b_new = zb - lr * gb
        t_new = (1 + math.sqrt(1 + 4 * t * t)) / 2
        zw = w_new + (t - 1) / t_new * (w_new - w)
        zb = b_new + (t - 1) / t_new * (b_new - b)
        w, b, t = w_new, b_new, t_new
    return w, b.item()


def rms_diff(sq_sum: Tensor, n_tokens: int) -> Tensor:
    """Safety Neurons' score: root mean squared activation difference between two models."""
    return (sq_sum / max(n_tokens, 1)).sqrt().float()
