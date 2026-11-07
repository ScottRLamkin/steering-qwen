"""Neuron steering: add a scaled ``down_proj`` column to a layer's MLP output via a forward hook.

Two ways to size the push, both from the original notebook:

* ``raw``  -- ``amount * c / ||c||``: a unit direction times a hand-picked amount
  (e.g. layer 14 neuron 8949 at 7.5 is "evil", layer 17 neuron 7426 at 7.5 is "poetic").
* ``dose`` -- ``dose * median_norm(L) * c / ||c||``: dose 1.0 pushes as hard as the length of a
  typical residual vector leaving layer ``L``, so doses are comparable across layers.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .generate import chat, chat_inputs
from .model import find_layers

Spec = list[tuple[int, int, float]]  # [(layer, neuron, amount_or_dose), ...]
Mode = Literal["raw", "dose"]

CALIB: list[str] = [
    "Tell me about your weekend.",
    "How do I bake bread?",
    "Explain photosynthesis briefly.",
    "What's a good name for a cat?",
]


class Steerer:
    def __init__(
        self,
        model: PreTrainedModel,
        tok: PreTrainedTokenizerBase,
        calib: list[str] | None = None,
    ) -> None:
        self.model = model
        self.tok = tok
        self.calib = calib if calib is not None else CALIB
        self.layers, self.layers_path = find_layers(model)
        self._handles: list = []
        self._active: dict[int, Tensor] = {}
        self._med: dict[int, float] = {}

    # ---- hooks -------------------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all steering hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._active.clear()

    def _add(self, L: int, v: Tensor) -> None:
        def hook(mod, inp, out):
            # .to() is a no-op once v matches; kept lazy so accelerate-offloaded layers work too
            return out + v.to(device=out.device, dtype=out.dtype)

        self._handles.append(self.layers[L].mlp.down_proj.register_forward_hook(hook))
        self._active[L] = v

    @property
    def active(self) -> bool:
        return bool(self._handles)

    # ---- directions and scales ---------------------------------------------------------------

    def _check(self, L: int, n: int) -> None:
        if not 0 <= L < len(self.layers):
            raise ValueError(f"layer {L} out of range (model has {len(self.layers)} layers)")
        width = self.layers[L].mlp.down_proj.in_features
        if not 0 <= n < width:
            raise ValueError(f"neuron {n} out of range (layer {L} MLP has {width} neurons)")

    @torch.no_grad()
    def down_col(self, L: int, n: int) -> Tensor:
        """Column ``n`` of layer ``L``'s ``down_proj`` as float32 ``[d_model]``.

        Computed by pushing a one-hot through the module so it also works for quantized
        (bitsandbytes) weights, where the column isn't directly readable. Calls ``.forward``
        directly so active steering hooks on this layer don't leak into the result.
        """
        self._check(L, n)
        dp = self.layers[L].mlp.down_proj
        device = dp.weight.device if dp.weight.device.type != "meta" else self.model.device  # meta = offloaded
        e = torch.zeros(1, dp.in_features, dtype=self.model.dtype, device=device)
        e[0, n] = 1.0
        return dp.forward(e)[0].float()

    @contextmanager
    def _unsteered(self) -> Iterator[None]:
        """Lift any active steering hooks for the duration, then put them back."""
        saved = dict(self._active)
        self.clear()
        try:
            yield
        finally:
            for layer, v in saved.items():
                self._add(layer, v)

    @torch.no_grad()
    def _layer_norms(self, input_ids: Tensor) -> list[Tensor]:
        """||residual leaving each layer|| per token, excluding position 0 (float32, CPU).

        Read with forward hooks on the decoder layers rather than ``output_hidden_states``:
        transformers replaces the last hidden state with the final-normed one.
        """
        norms: list[Tensor] = [torch.empty(0)] * len(self.layers)

        def recorder(i: int):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                norms[i] = h[0, 1:].float().norm(dim=-1).cpu()

            return hook

        handles = [layer.register_forward_hook(recorder(i)) for i, layer in enumerate(self.layers)]
        try:
            self.model(input_ids=input_ids.to(self.model.device), use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        return norms

    @torch.no_grad()
    def median_norm(self, L: int) -> float:
        """Median ||residual leaving layer L|| over the calibration prompts (excluding position 0).

        Always measured on the unsteered model: active hooks are lifted while measuring.
        Uses the values from :meth:`calibrate_on_texts` if that was called.
        """
        if L in self._med:
            return self._med[L]
        with self._unsteered():
            per_layer: list[list[Tensor]] = [[] for _ in self.layers]
            for t in self.calib:
                ids = chat_inputs(self.tok, t, self.model.device)["input_ids"]
                for i, n in enumerate(self._layer_norms(ids)):
                    per_layer[i].append(n)
            for i, ns in enumerate(per_layer):
                self._med.setdefault(i, torch.cat(ns).median().item())
        return self._med[L]

    @torch.no_grad()
    def calibrate_on_texts(self, texts: list[str]) -> dict[int, float]:
        """Set every layer's median residual norm from raw texts, e.g. chat-formatted question+answer.

        This is HedwigAI's calibration: tokens 1: of each text, median taken as
        ``sorted(v)[len(v) // 2]`` (the upper median, as in their ``mx.sort(v)[v.size // 2]``).
        Replaces any cached medians, so :meth:`set_dosed` uses these from now on.
        """
        with self._unsteered():
            per_layer: list[list[Tensor]] = [[] for _ in self.layers]
            for text in texts:
                ids = torch.tensor([self.tok.encode(text)])
                for i, n in enumerate(self._layer_norms(ids)):
                    per_layer[i].append(n)
        self._med = {}
        for i, ns in enumerate(per_layer):
            v = torch.cat(ns).sort().values
            self._med[i] = v[v.numel() // 2].item()
        return dict(self._med)

    @property
    def medians(self) -> dict[int, float]:
        """The median residual norms cached so far, by layer."""
        return dict(self._med)

    def use_medians(self, med: dict[int, float]) -> None:
        """Use previously measured median norms (e.g. saved by :meth:`calibrate_on_texts`)."""
        self._med = {int(k): float(v) for k, v in med.items()}

    @torch.no_grad()
    def down_cols(self, L: int, chunk: int = 1024) -> Tensor:
        """Every column of layer ``L``'s ``down_proj`` as float32 ``[d_model, d_ff]``.

        Same one-hot trick as :meth:`down_col`, in chunks, so it also works for bitsandbytes weights.
        """
        dp = self.layers[L].mlp.down_proj
        device = dp.weight.device if dp.weight.device.type != "meta" else self.model.device
        out = []
        for s in range(0, dp.in_features, chunk):
            n = min(chunk, dp.in_features - s)
            e = torch.zeros(n, dp.in_features, dtype=self.model.dtype, device=device)
            e[torch.arange(n), torch.arange(s, s + n)] = 1.0
            out.append(dp.forward(e).float())
        return torch.cat(out).T.contiguous()

    def vectors(self, spec: Spec, mode: Mode = "raw", verbose: bool = False) -> dict[int, Tensor]:
        """Sum the steering vectors for ``spec`` per layer (float32)."""
        add: dict[int, Tensor] = {}
        for L, n, x in spec:
            c = self.down_col(L, n)
            scale = x / c.norm() if mode == "raw" else x * self.median_norm(L) / c.norm()
            v = scale * c
            if verbose:
                print(
                    f"  L{L} n{n} {mode}={x:+g}: ||col||={c.norm().item():.4f} ||push||={v.norm().item():.3f}",
                    file=sys.stderr,
                )
            add[L] = add[L] + v if L in add else v
        return add

    # ---- public steering API -----------------------------------------------------------------

    def set(self, spec: Spec, mode: Mode = "raw", verbose: bool = False) -> None:
        """Replace any current steering with ``spec``."""
        self.clear()
        for L, v in self.vectors(spec, mode, verbose).items():
            self._add(L, v)

    def set_raw(self, spec: Spec, verbose: bool = False) -> None:
        self.set(spec, "raw", verbose)

    def set_dosed(self, spec: Spec, verbose: bool = False) -> None:
        self.set(spec, "dose", verbose)

    @contextmanager
    def steering(self, spec: Spec, mode: Mode = "raw", verbose: bool = False) -> Iterator["Steerer"]:
        """``with st.steering([(14, 8949, 7.5)]): ...`` -- hooks are removed on exit."""
        self.set(spec, mode, verbose)
        try:
            yield self
        finally:
            self.clear()

    def generate(self, prompt: str, max_new_tokens: int = 150, sample: bool = False, stream: bool = False) -> str:
        """Generate with whatever steering is currently active."""
        return chat(self.model, self.tok, prompt, max_new_tokens=max_new_tokens, sample=sample, stream=stream)
