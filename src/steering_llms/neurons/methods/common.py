"""What every recipe shares: the loaded model, its run directory, caching and logging."""

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ...model import find_layers, quant_kind
from ..intervene import Edit
from ..neuronset import NeuronSet, load_sets, save_sets


@dataclass
class Ctx:
    model: PreTrainedModel
    tok: PreTrainedTokenizerBase
    run_dir: Path
    layers: nn.ModuleList = field(init=False)
    quiet: bool = False

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.layers = find_layers(self.model)[0]
        self._t0 = time.time()

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def width(self) -> int:
        return self.layers[0].mlp.down_proj.in_features

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self._t0:7.0f}s] {msg}"
        if not self.quiet:
            print(line, file=sys.stderr, flush=True)
        with open(self.run_dir / "log.txt", "a") as f:
            f.write(line + "\n")

    def path(self, name: str) -> Path:
        p = self.run_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def cached_json(self, name: str, fn: Callable[[], Any]) -> Any:
        """Load ``name`` from the run directory, or compute it with ``fn`` and save it."""
        p = self.path(name)
        if p.exists():
            return json.loads(p.read_text())
        out = fn()
        p.write_text(json.dumps(out, indent=1))
        return out

    def cached_sets(self, name: str, fn: Callable[[], dict[str, NeuronSet]]) -> dict[str, NeuronSet]:
        p = self.path(name)
        if p.exists():
            return load_sets(p)
        sets = fn()
        save_sets(sets, p)
        return sets

    def save_json(self, name: str, obj: Any) -> None:
        self.path(name).write_text(json.dumps(obj, indent=1))

    def save_edits(self, name: str, edits: list[Edit]) -> None:
        self.save_json(name, [e.to_dict() for e in edits])


def model_info(model: PreTrainedModel) -> dict:
    q = getattr(model, "quantization_config", None) or getattr(model.config, "quantization_config", None)
    return {
        "model": model.config._name_or_path if hasattr(model.config, "_name_or_path") else "",
        "dtype": str(model.dtype),
        "quantized": bool(q),
        "quant": quant_kind(model),
        "device": str(model.device),
        "torch": torch.__version__,
    }


def set_summary(sets: dict[str, NeuronSet]) -> dict:
    return {k: {"n": len(v), "site": v.site.value, "layers": v.layer_counts()} for k, v in sets.items()}
