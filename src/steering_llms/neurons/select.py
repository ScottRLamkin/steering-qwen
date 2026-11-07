"""Turn per-neuron score tensors into :class:`NeuronSet` s: top-k, thresholds, quantiles, caps,
consensus across prompts and the "universal neuron" filter.

Scores are ``[n_layers_read, width]`` tensors whose row ``j`` is layer ``layer_ids[j]``; masks are
bool tensors of the same shape.
"""

from collections import Counter

import torch
from torch import Tensor

from .neuronset import NeuronSet
from .sites import Site
from .stats import flat_quantile


def _key(score: Tensor, by_abs: bool) -> Tensor:
    return score.abs() if by_abs else score


def top_k_mask(score: Tensor, k: int, by_abs: bool = True, per_layer: bool = False) -> Tensor:
    """The ``k`` largest scores, globally or in every layer."""
    key = _key(score, by_abs).float()
    mask = torch.zeros_like(key, dtype=torch.bool)
    if per_layer:
        idx = key.topk(min(k, key.shape[1]), dim=1).indices
        mask.scatter_(1, idx, True)
    else:
        flat = key.reshape(-1)
        mask.view(-1)[flat.topk(min(k, flat.numel())).indices] = True
    return mask


def threshold_mask(score: Tensor, thr: float, by_abs: bool = True, below: bool = False) -> Tensor:
    key = _key(score, by_abs)
    return key < thr if below else key > thr


def quantile_mask(score: Tensor, q: float, by_abs: bool = True, per_layer: bool = True) -> Tensor:
    """Scores above their layer's (or the global) ``q``-quantile, e.g. q=0.995 for DPN-LE."""
    key = _key(score, by_abs).float()
    if per_layer:
        thr = torch.stack([torch.tensor(flat_quantile(row, q)) for row in key])[:, None]
        return key > thr
    return key > flat_quantile(key, q)


def bottom_quantile_mask(score: Tensor, q: float, among: Tensor | None = None) -> Tensor:
    """Scores at or below the global ``q``-quantile (e.g. the lowest 1% of entropy). With ``among``,
    the quantile is taken over those entries only and nothing outside them is kept."""
    if among is None:
        return score <= flat_quantile(score, q)
    if not among.any():
        return torch.zeros_like(among)
    return among & (score <= flat_quantile(score[among], q))


def cap_per_layer(mask: Tensor, score: Tensor, cap: int, by_abs: bool = True) -> Tensor:
    """Keep at most ``cap`` selected neurons per layer, the highest-scoring ones."""
    key = _key(score, by_abs).float().masked_fill(~mask, float("-inf"))
    out = torch.zeros_like(mask)
    k = min(cap, mask.shape[1])
    vals, idx = key.topk(k, dim=1)
    out.scatter_(1, idx, vals > float("-inf"))
    return out & mask


def cap_total(mask: Tensor, score: Tensor, cap: int, by_abs: bool = True) -> Tensor:
    key = _key(score, by_abs).float().masked_fill(~mask, float("-inf"))
    out = torch.zeros_like(mask)
    flat = key.reshape(-1)
    vals, idx = flat.topk(min(cap, flat.numel()))
    out.view(-1)[idx[vals > float("-inf")]] = True
    return out


def to_set(
    mask: Tensor,
    score: Tensor,
    layer_ids: list[int],
    site: Site,
    extras: dict[str, Tensor] | None = None,
    label: str = "",
    meta: dict | None = None,
    sort_by_abs: bool = True,
) -> NeuronSet:
    """A NeuronSet of the masked neurons, ordered by score (largest |score| first)."""
    rows, cols = mask.nonzero(as_tuple=True)
    sc = score[rows, cols].float()
    order = (sc.abs() if sort_by_abs else sc).argsort(descending=True)
    rows, cols, sc = rows[order], cols[order], sc[order]
    ex = {k: v[rows, cols].float().tolist() for k, v in (extras or {}).items()}
    return NeuronSet(
        Site(site),
        [layer_ids[r] for r in rows.tolist()],
        cols.tolist(),
        sc.tolist(),
        ex,
        label,
        meta or {},
    )


def to_mask(ns: NeuronSet, layer_ids: list[int], width: int) -> Tensor:
    pos = {L: j for j, L in enumerate(layer_ids)}
    m = torch.zeros(len(layer_ids), width, dtype=torch.bool)
    for L, n in ns.keys():
        if L in pos:
            m[pos[L], n] = True
    return m


def consensus(sets: list[NeuronSet], min_frac: float) -> list[tuple[int, int]]:
    """Neurons present in more than ``min_frac`` of ``sets`` (Knowledge Neurons' p% filter)."""
    c = Counter(k for s in sets for k in set(s.keys()))
    return [k for k, v in c.items() if v > min_frac * len(sets)]


def frequent(masks: list[Tensor], min_frac: float) -> Tensor:
    """Neurons selected in at least ``min_frac`` of ``masks``: Contrastive Pair Search's
    "universal" neurons, active on almost any prompt, which are removed before steering."""
    return torch.stack(masks).float().mean(0) >= min_frac


def overlap_table(sets: dict[str, NeuronSet]) -> dict[str, dict[str, float]]:
    """Pairwise Jaccard overlap of named sets."""
    names = list(sets)
    return {a: {b: sets[a].jaccard(sets[b]) for b in names} for a in names}
