"""One way to change neuron activity for every method: a list of :class:`Edit` applied by hooks.

Each edit picks neurons at one site of one layer and applies an operation to them:

* ``mul``        ``x * v``      scale; 0 ablates (Contrastive Pair Search, AurA, Knowledge Neurons)
* ``set``        ``v``          clamp to a value (Psychological Concept Neurons, sentiment unit)
* ``add``        ``x + v``      shift (DPN-LE, NPTI, Emotion & Rhetoric steering)
* ``clamp_max``  ``min(x, v)``  NPTI's ``min(0, n)`` for opposite-pole neurons
* ``clamp_min``  ``max(x, v)``

``v`` is a scalar or one value per neuron. ``value_fn`` supplies values at run time instead
(dynamic activation patching from another model).

``scope`` picks positions: ``all`` (every token, prompt and generated), ``decode`` (only
generation steps after the prompt, as NPTI's and DPN-LE's code do), ``prefill_last`` (the last
prompt token, i.e. the step that produces the first generated token), ``last`` (the last
position of every call). ``start``/``end`` limit ``all`` to positions ``start:end`` of a teacher-forced
pass (cached single-token steps are always edited).
``keep_prob`` < 1 re-draws, on every forward call, which neurons are edited (NPTI keeps 90%).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn

from .neuronset import NeuronSet
from .sites import Site, check_layers, hook_site

Op = Literal["mul", "set", "add", "clamp_max", "clamp_min"]
Scope = Literal["all", "decode", "prefill_last", "last"]
OPS = ("mul", "set", "add", "clamp_max", "clamp_min")


@dataclass
class Edit:
    layer: int
    index: list[int]
    op: Op
    value: float | list[float] = 0.0
    site: Site = Site.DOWN_IN
    scope: Scope = "all"
    start: int = 0
    keep_prob: float = 1.0
    end: int | None = None
    value_fn: Callable[[Tensor], Tensor] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.site = Site(self.site)
        if self.op not in OPS:
            raise ValueError(f"unknown op {self.op!r}")
        if isinstance(self.value, list) and len(self.value) != len(self.index):
            raise ValueError("per-neuron values must match the number of neurons")

    def to_dict(self) -> dict:
        if self.value_fn is not None:
            raise ValueError("an edit with value_fn can't be serialized")
        return {
            "layer": self.layer,
            "index": list(self.index),
            "op": self.op,
            "value": self.value,
            "site": self.site.value,
            "scope": self.scope,
            "start": self.start,
            "keep_prob": self.keep_prob,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Edit":
        return cls(**d)


def edits_from_set(
    ns: NeuronSet,
    op: Op,
    value: float | str | Callable[[dict[str, float]], float] = 0.0,
    scope: Scope = "all",
    keep_prob: float = 1.0,
) -> list[Edit]:
    """One edit per layer of ``ns``. ``value`` is a constant, the name of a column (``"score"`` or an
    extra), or a function of a row ``{"score": ..., extra: ...}``."""
    by: dict[int, tuple[list[int], list[float]]] = {}
    for i, (L, n) in enumerate(zip(ns.layer, ns.index)):
        row = {"score": ns.score[i], **{k: v[i] for k, v in ns.extras.items()}}
        if callable(value):
            v = float(value(row))
        elif isinstance(value, str):
            v = float(row[value])
        else:
            v = float(value)
        idx, vals = by.setdefault(L, ([], []))
        idx.append(n)
        vals.append(v)
    out = []
    for L, (idx, vals) in sorted(by.items()):
        v: float | list[float] = vals[0] if not callable(value) and not isinstance(value, str) else vals
        out.append(Edit(L, idx, op, v, ns.site, scope, keep_prob=keep_prob))
    return out


def _positions(T: int, scope: Scope, start: int, end: int | None = None) -> slice | None:
    if scope == "all":
        return slice(start, end) if T > 1 else slice(0, None)
    if scope == "decode":
        return slice(None) if T == 1 else None
    if scope == "prefill_last":
        return slice(T - 1, T) if T > 1 else None
    if scope == "last":
        return slice(T - 1, T)
    raise ValueError(f"unknown scope {scope!r}")


def _apply(op: Op, x: Tensor, v: Tensor) -> Tensor:
    if op == "mul":
        return x * v
    if op == "set":
        return v.expand_as(x)
    if op == "add":
        return x + v
    if op == "clamp_max":
        return torch.minimum(x, v)
    return torch.maximum(x, v)


class Intervention:
    """Apply edits with forward hooks. Use as a context manager, or call :meth:`apply`/:meth:`remove`."""

    def __init__(self, layers: nn.ModuleList, edits: list[Edit], seed: int = 0) -> None:
        check_layers(layers)
        self.layers = layers
        self.edits = list(edits)
        self.handles: list = []
        self.gen = torch.Generator().manual_seed(seed)
        self._cache: dict[tuple[int, str], tuple[Tensor, Tensor]] = {}

    def _tensors(self, i: int, e: Edit, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        key = (i, f"{device}{dtype}")
        if key not in self._cache:
            idx = torch.tensor(e.index, dtype=torch.long, device=device)
            v = torch.tensor(e.value if isinstance(e.value, list) else [e.value], dtype=dtype, device=device)
            self._cache[key] = (idx, v)
        return self._cache[key]

    def _hook_for(self, items: list[tuple[int, Edit]]):
        def fn(x: Tensor) -> Tensor | None:
            out = None
            T = x.shape[1]
            for i, e in items:
                pos = _positions(T, e.scope, e.start, e.end)
                if pos is None:
                    continue
                idx, v = self._tensors(i, e, x.device, x.dtype)
                if e.keep_prob < 1.0:
                    keep = (torch.rand(len(e.index), generator=self.gen) < e.keep_prob).to(x.device)
                    idx = idx[keep]
                    v = v[keep] if v.numel() > 1 else v
                    if idx.numel() == 0:
                        continue
                if out is None:
                    out = x.clone()
                view = out[:, pos]
                cur = view[..., idx]
                if e.value_fn is not None:
                    v = e.value_fn(cur).to(device=x.device, dtype=x.dtype)
                view[..., idx] = _apply(e.op, cur, v)
            return out

        return fn

    def apply(self) -> "Intervention":
        if self.handles:
            return self
        groups: dict[tuple[int, Site], list[tuple[int, Edit]]] = {}
        for i, e in enumerate(self.edits):
            groups.setdefault((e.layer, e.site), []).append((i, e))
        for (L, site), items in groups.items():
            self.handles.append(hook_site(self.layers[L], site, self._hook_for(items)))
        return self

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def __enter__(self) -> "Intervention":
        return self.apply()

    def __exit__(self, *exc) -> None:
        self.remove()

    @property
    def active(self) -> bool:
        return bool(self.handles)


def save_edits(edits: list[Edit], path) -> None:
    import json
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([e.to_dict() for e in edits]))


def load_edits(path) -> list[Edit]:
    import json
    from pathlib import Path

    return [Edit.from_dict(d) for d in json.loads(Path(path).read_text())]
