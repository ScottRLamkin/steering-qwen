"""Read neuron activations over many texts and keep per-group statistics.

Every "find by forward passes" method needs the same few numbers per neuron and per group of
texts (a trait pole, a style, a label): token count, sum, sum of squares, how often it is > 0,
its maximum, optionally a histogram (for percentiles such as NPTI's a95), and optionally one
pooled value per text (for AUROC, average precision, Cohen's d over samples, probes).

A :class:`Sample` is token ids plus the span of positions that count (``start:end``), e.g. the
last prompt token, the answer, or every token after the first. Texts are batched with right
padding, which never reaches a counted position.
"""

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..discovery import chat_text
from .sites import Site, check_layers, hook_site, site_width

Pool = Literal["mean", "max", "last"]


@dataclass
class Sample:
    ids: list[int]
    start: int = 0
    end: int | None = None  # exclusive; None = end of sequence
    group: str = ""

    def span(self) -> tuple[int, int]:
        end = len(self.ids) if self.end is None else self.end
        return max(self.start, 0), min(end, len(self.ids))


# ---- building samples ------------------------------------------------------------------------------


def text_sample(tok: PreTrainedTokenizerBase, text: str, group: str = "", skip_first: bool = True) -> Sample:
    """Raw text with no chat template, every token (but the first, the attention sink) counted."""
    ids = tok.encode(text)
    return Sample(ids, 1 if skip_first and len(ids) > 1 else 0, None, group)


def prompt_sample(
    tok: PreTrainedTokenizerBase, prompt: str, group: str = "", system: str | None = None, chat: bool = True
) -> Sample:
    """A chat-formatted prompt (thinking off), counting only its last token."""
    ids = tok.encode(chat_text(tok, prompt, system) if chat else prompt)
    return Sample(ids, len(ids) - 1, len(ids), group)


def answer_sample(
    tok: PreTrainedTokenizerBase,
    prompt: str,
    answer: str,
    group: str = "",
    system: str | None = None,
    max_answer_tokens: int | None = None,
) -> Sample:
    """Chat prompt + answer text re-read teacher-forced, counting the answer tokens only."""
    pre_text = chat_text(tok, prompt, system)
    pre = len(tok.encode(pre_text))
    ids = tok.encode(pre_text + answer)
    end = len(ids) if max_answer_tokens is None else min(len(ids), pre + max_answer_tokens)
    return Sample(ids[:end], pre, end, group)


# ---- statistics ------------------------------------------------------------------------------------


@dataclass
class GroupStats:
    """Per-neuron statistics of one group, ``[n_layers_read, width]`` tensors on CPU."""

    layers: list[int]
    n_tokens: int = 0
    n_samples: int = 0
    sum: Tensor | None = None  # float64
    sumsq: Tensor | None = None  # float64
    pos: Tensor | None = None  # float64 count of x > 0
    max: Tensor | None = None  # float32
    hist: Tensor | None = None  # float32 counts [L, W, bins]
    hist_range: tuple[float, float, int] | None = None
    samples: list[Tensor] = field(default_factory=list)  # pooled per sample, float16 [L, W] each

    @property
    def mean(self) -> Tensor:
        return (self.sum / max(self.n_tokens, 1)).float()

    @property
    def var(self) -> Tensor:
        m = self.sum / max(self.n_tokens, 1)
        return (self.sumsq / max(self.n_tokens, 1) - m * m).clamp_min(0).float()

    @property
    def frac_pos(self) -> Tensor:
        return (self.pos / max(self.n_tokens, 1)).float()

    def sample_matrix(self) -> Tensor:
        """Pooled values, ``[n_samples, n_layers_read, width]`` float32."""
        return torch.stack(self.samples).float()

    def percentile(self, q: float) -> Tensor:
        """The q-th percentile (0-100) per neuron from the histogram (upper bin edge)."""
        if self.hist is None or self.hist_range is None:
            raise RuntimeError("collect with hist=(lo, hi, bins) to get percentiles")
        lo, hi, bins = self.hist_range
        cdf = self.hist.cumsum(-1)
        total = cdf[..., -1:].clamp_min(1)
        b = (cdf < q / 100 * total).sum(-1).clamp(max=bins - 1)
        return lo + (b.float() + 1) * (hi - lo) / bins


def merge_stats(*groups: GroupStats) -> GroupStats:
    """Pool groups' token-level moments (e.g. "the other genders" for a one-vs-rest contrast)."""
    out = GroupStats(groups[0].layers, hist_range=groups[0].hist_range)
    for g in groups:
        out.n_tokens += g.n_tokens
        out.n_samples += g.n_samples
        for k in ("sum", "sumsq", "pos", "hist"):
            v = getattr(g, k)
            if v is not None:
                cur = getattr(out, k)
                setattr(out, k, v.clone() if cur is None else cur + v)
        if g.max is not None:
            out.max = g.max.clone() if out.max is None else torch.maximum(out.max, g.max)
        out.samples.extend(g.samples)
    return out


class Collector:
    """Hooks one site on the chosen layers and accumulates :class:`GroupStats` per group."""

    def __init__(
        self,
        layers: nn.ModuleList,
        site: Site = Site.DOWN_IN,
        layer_ids: Iterable[int] | None = None,
        pool: Pool | None = None,
        hist: tuple[float, float, int] | None = None,
        moments: bool = True,
    ) -> None:
        check_layers(layers)
        self.layers = layers
        self.site = Site(site)
        self.layer_ids = list(layer_ids) if layer_ids is not None else list(range(len(layers)))
        self.pool = pool
        self.hist = hist
        self.moments = moments

    @torch.no_grad()
    def run(
        self,
        model: PreTrainedModel,
        samples: list[Sample],
        batch_size: int = 8,
        log: Callable[[str], None] | None = None,
        stats: dict[str, GroupStats] | None = None,
    ) -> dict[str, GroupStats]:
        """Forward every sample once and return ``{group: GroupStats}`` (updates ``stats`` if given)."""
        stats = stats if stats is not None else {}
        width = site_width(self.layers[0], self.site)
        by_group: dict[str, list[Sample]] = {}
        for s in samples:
            by_group.setdefault(s.group, []).append(s)
        t0, done = time.time(), 0
        for group, items in by_group.items():
            st = stats.setdefault(group, GroupStats(self.layer_ids, hist_range=self.hist))
            for b in range(0, len(items), batch_size):
                batch = items[b : b + batch_size]
                self._batch(model, batch, st, width)
                done += len(batch)
            if log:
                log(f"collected {group}: {st.n_samples} samples, {st.n_tokens} tokens ({time.time() - t0:.0f}s)")
        return stats

    def _batch(self, model: PreTrainedModel, batch: list[Sample], st: GroupStats, width: int) -> None:
        dev = model.device
        T = max(len(s.ids) for s in batch)
        ids = torch.tensor([s.ids + [0] * (T - len(s.ids)) for s in batch], device=dev)
        mask = torch.zeros(len(batch), T, dtype=torch.bool, device=dev)
        for r, s in enumerate(batch):
            a, e = s.span()
            mask[r, a:e] = True
        n_tok = int(mask.sum())
        nl = len(self.layer_ids)
        acc: dict[str, list] = {k: [None] * nl for k in ("sum", "sumsq", "pos", "max", "hist", "pooled")}

        def reader(j: int):
            def fn(x: Tensor) -> None:
                xf = x.float()
                m = mask[..., None]
                if self.moments:
                    xm = xf * m
                    acc["sum"][j] = xm.sum((0, 1)).cpu().double()
                    acc["sumsq"][j] = (xm * xf).sum((0, 1)).cpu().double()
                    acc["pos"][j] = ((xf > 0) & m).sum((0, 1)).cpu().double()
                    acc["max"][j] = xf.masked_fill(~m, float("-inf")).amax((0, 1)).cpu()
                if self.hist is not None:
                    lo, hi, bins = self.hist
                    v = xf[mask]  # [tokens, W]
                    bucket = ((v - lo) / (hi - lo) * bins).floor().clamp(0, bins - 1).long()
                    counts = torch.zeros(width, bins, device=x.device)
                    counts.scatter_add_(1, bucket.T.contiguous(), torch.ones_like(bucket.T, dtype=torch.float32))
                    acc["hist"][j] = counts.cpu()
                if self.pool is not None:
                    if self.pool == "mean":
                        p = (xf * m).sum(1) / m.sum(1).clamp_min(1)
                    elif self.pool == "max":
                        p = xf.masked_fill(~m, float("-inf")).amax(1)
                    else:
                        last = torch.tensor([s.span()[1] - 1 for s in batch], device=x.device)
                        p = xf[torch.arange(len(batch), device=x.device), last]
                    acc["pooled"][j] = p.half().cpu()
                return None

            return fn

        handles = [hook_site(self.layers[L], self.site, reader(j)) for j, L in enumerate(self.layer_ids)]
        try:
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()

        st.n_tokens += n_tok
        st.n_samples += len(batch)
        if self.moments:
            for k in ("sum", "sumsq", "pos"):
                new = torch.stack(acc[k])
                cur = getattr(st, k)
                setattr(st, k, new if cur is None else cur + new)
            new_max = torch.stack(acc["max"])
            st.max = new_max if st.max is None else torch.maximum(st.max, new_max)
        if self.hist is not None:
            new = torch.stack(acc["hist"])
            st.hist = new if st.hist is None else st.hist + new
        if self.pool is not None:
            pooled = torch.stack(acc["pooled"], 1)  # [B, L, W]
            st.samples.extend(pooled.unbind(0))


@torch.no_grad()
def paired_sq_diff(
    model_a: PreTrainedModel,
    layers_a: nn.ModuleList,
    model_b: PreTrainedModel,
    layers_b: nn.ModuleList,
    samples: list[Sample],
    site: Site = Site.DOWN_IN,
    batch_size: int = 2,
    log: Callable[[str], None] | None = None,
) -> tuple[Tensor, int]:
    """Sum over counted positions of ``(a_A - a_B)^2`` per neuron, for the same texts in two models.

    Returns ``(sum [n_layers, width] float64, n_tokens)``; Safety Neurons' score is
    ``sqrt(sum / n_tokens)``. Model A's activations are held on its device for one batch only.
    """
    site = Site(site)
    n = len(layers_a)
    total: Tensor | None = None
    n_tok = 0
    t0 = time.time()
    for b in range(0, len(samples), batch_size):
        batch = samples[b : b + batch_size]
        T = max(len(s.ids) for s in batch)
        ids = [s.ids + [0] * (T - len(s.ids)) for s in batch]
        mask = torch.zeros(len(batch), T, dtype=torch.bool)
        for r, s in enumerate(batch):
            a, e = s.span()
            mask[r, a:e] = True
        held: list[Tensor | None] = [None] * n
        out: list[Tensor | None] = [None] * n

        def keep(j: int):
            def fn(x: Tensor) -> None:
                held[j] = x[mask.to(x.device)].float()
                return None

            return fn

        def diff(j: int):
            def fn(x: Tensor) -> None:
                d = x[mask.to(x.device)].float() - held[j].to(x.device)
                out[j] = (d * d).sum(0).cpu().double()
                return None

            return fn

        hs = [hook_site(layers_a[j], site, keep(j)) for j in range(n)]
        try:
            model_a(input_ids=torch.tensor(ids, device=model_a.device), use_cache=False, logits_to_keep=1)
        finally:
            for h in hs:
                h.remove()
        hs = [hook_site(layers_b[j], site, diff(j)) for j in range(n)]
        try:
            model_b(input_ids=torch.tensor(ids, device=model_b.device), use_cache=False, logits_to_keep=1)
        finally:
            for h in hs:
                h.remove()
        s = torch.stack(out)
        total = s if total is None else total + s
        n_tok += int(mask.sum())
        if log and (b // batch_size) % 10 == 0:
            log(f"paired diff {b + len(batch)}/{len(samples)} ({time.time() - t0:.0f}s)")
    return total, n_tok
