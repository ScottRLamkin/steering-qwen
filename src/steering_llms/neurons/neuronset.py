"""A set of neurons with scores and per-neuron extras, saved as JSON with its provenance."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from .sites import Site


@dataclass
class NeuronSet:
    """Columnar: ``layer[i]``, ``index[i]``, ``score[i]`` and ``extras[name][i]`` describe neuron i.

    ``label`` names what the set is for (e.g. ``"Extraversion+"``); ``meta`` records the method,
    its settings and anything needed to trace where a neuron came from.
    """

    site: Site
    layer: list[int] = field(default_factory=list)
    index: list[int] = field(default_factory=list)
    score: list[float] = field(default_factory=list)
    extras: dict[str, list[float]] = field(default_factory=dict)
    label: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.site = Site(self.site)
        n = len(self.layer)
        if len(self.index) != n or len(self.score) != n or any(len(v) != n for v in self.extras.values()):
            raise ValueError("NeuronSet columns must have equal length")

    def __len__(self) -> int:
        return len(self.layer)

    def keys(self) -> list[tuple[int, int]]:
        return list(zip(self.layer, self.index))

    def by_layer(self) -> dict[int, dict[str, Tensor]]:
        """``{layer: {"index": LongTensor, "score": Tensor, extra: Tensor, ...}}``."""
        out: dict[int, dict[str, list]] = {}
        for i, L in enumerate(self.layer):
            d = out.setdefault(L, {"index": [], "score": [], **{k: [] for k in self.extras}})
            d["index"].append(self.index[i])
            d["score"].append(self.score[i])
            for k, v in self.extras.items():
                d[k].append(v[i])
        return {
            L: {k: torch.tensor(v, dtype=torch.long if k == "index" else torch.float32) for k, v in d.items()}
            for L, d in sorted(out.items())
        }

    def subset(self, keep: list[int]) -> "NeuronSet":
        return NeuronSet(
            self.site,
            [self.layer[i] for i in keep],
            [self.index[i] for i in keep],
            [self.score[i] for i in keep],
            {k: [v[i] for i in keep] for k, v in self.extras.items()},
            self.label,
            dict(self.meta),
        )

    def top(self, k: int, by_abs: bool = True) -> "NeuronSet":
        order = sorted(range(len(self)), key=lambda i: abs(self.score[i]) if by_abs else self.score[i], reverse=True)
        return self.subset(order[:k])

    def minus(self, other: "NeuronSet") -> "NeuronSet":
        drop = set(other.keys())
        return self.subset([i for i, k in enumerate(self.keys()) if k not in drop])

    def jaccard(self, other: "NeuronSet") -> float:
        a, b = set(self.keys()), set(other.keys())
        return len(a & b) / len(a | b) if a | b else 0.0

    def layer_counts(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for L in self.layer:
            out[L] = out.get(L, 0) + 1
        return dict(sorted(out.items()))

    def to_dict(self) -> dict:
        return {
            "site": self.site.value,
            "label": self.label,
            "meta": self.meta,
            "layer": self.layer,
            "index": self.index,
            "score": self.score,
            "extras": self.extras,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NeuronSet":
        return cls(d["site"], d["layer"], d["index"], d["score"], d.get("extras", {}), d.get("label", ""), d.get("meta", {}))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "NeuronSet":
        return cls.from_dict(json.loads(Path(path).read_text()))


def save_sets(sets: dict[str, NeuronSet], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({k: v.to_dict() for k, v in sets.items()}))


def load_sets(path: str | Path) -> dict[str, NeuronSet]:
    return {k: NeuronSet.from_dict(v) for k, v in json.loads(Path(path).read_text()).items()}
