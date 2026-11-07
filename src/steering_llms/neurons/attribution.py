"""Gradient-based neuron attribution and the exact patching that checks it.

* :func:`atp`                   Attribution Patching (AtP, Kramár et al. 2403.00745): for a clean and a
                                token-aligned corrupted prompt, ``(a_noise - a_clean) * dM/da`` at
                                every (position, neuron) estimates the metric change from patching
                                that neuron's corrupted value into the clean run.
* :func:`exact_patch`           the real thing, batched: one forward per candidate, verifying AtP.
* :func:`integrated_gradients`  Knowledge Neurons' attribution (Dai et al. 2104.08696): scale one
                                layer's activation at one position from 0 to its value and
                                integrate the gradient of the answer probability.

HedwigAI's search (``discovery.attribution_grads``) is AtP's linear estimate with
``delta = dose * down_proj column`` added to the residual instead of a patch.

A *metric* maps the last position's logits ``[B, vocab]`` to one number per row ``[B]``.
"""

import time
from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from ..model import find_layers
from .sites import Site, check_layers, hook_site, site_width

Metric = Callable[[Tensor], Tensor]


def logprob_metric(token: int) -> Metric:
    return lambda logits: logits.float().log_softmax(-1)[:, token]


def prob_metric(token: int) -> Metric:
    return lambda logits: logits.float().softmax(-1)[:, token]


def logit_diff_metric(good: int, bad: int) -> Metric:
    return lambda logits: logits[:, good].float() - logits[:, bad].float()


@torch.no_grad()
def read_site(
    model: PreTrainedModel, layers: nn.ModuleList, ids: Tensor, site: Site, layer_ids: Iterable[int]
) -> dict[int, Tensor]:
    """``{layer: activation [B, T, width]}`` at ``site`` for ``ids`` (no edits)."""
    acts: dict[int, Tensor] = {}

    def rec(L: int):
        def fn(x: Tensor) -> None:
            acts[L] = x.detach().clone()
            return None

        return fn

    hs = [hook_site(layers[L], site, rec(L)) for L in layer_ids]
    try:
        model(input_ids=ids.to(model.device), use_cache=False, logits_to_keep=1)
    finally:
        for h in hs:
            h.remove()
    return acts


def atp(
    model: PreTrainedModel,
    layers: nn.ModuleList,
    clean: Tensor,
    noise: Tensor,
    metric: Metric,
    site: Site = Site.DOWN_IN,
    layer_ids: Iterable[int] | None = None,
) -> tuple[Tensor, float]:
    """AtP estimates for one clean/corrupted pair, ``[n_layers_read, T, width]`` float32 on CPU.

    ``clean`` and ``noise`` are ``[1, T]`` token ids of equal length. Returns the estimates and the
    clean metric. The estimate at (L, t, n) approximates ``M(clean with a_Ltn := noise) - M(clean)``.
    """
    check_layers(layers)
    site = Site(site)
    layer_ids = list(layer_ids) if layer_ids is not None else list(range(len(layers)))
    if clean.shape != noise.shape:
        raise ValueError("AtP needs token-aligned clean and corrupted prompts of equal length")
    a_noise = read_site(model, layers, noise, site, layer_ids)
    a_clean: dict[int, Tensor] = {}
    deltas: dict[int, Tensor] = {}

    def add(L: int):
        def fn(x: Tensor) -> Tensor:
            a_clean[L] = x.detach()
            d = torch.zeros(x.shape, dtype=torch.float32, device=x.device, requires_grad=True)
            deltas[L] = d
            return x + d.to(x.dtype)

        return fn

    grads_were = {n: p.requires_grad for n, p in model.named_parameters()}
    model.requires_grad_(False)
    hs = [hook_site(layers[L], site, add(L)) for L in layer_ids]
    try:
        with torch.enable_grad():
            logits = model(input_ids=clean.to(model.device), use_cache=False, logits_to_keep=1).logits[:, -1]
            m = metric(logits).sum()
            g = torch.autograd.grad(m, [deltas[L] for L in layer_ids])
    finally:
        for h in hs:
            h.remove()
        for n, p in model.named_parameters():
            p.requires_grad_(grads_were[n])
    out = torch.stack(
        [((a_noise[L].float() - a_clean[L].float()) * gi.float())[0].cpu() for L, gi in zip(layer_ids, g)]
    )
    return out, m.item()


@torch.no_grad()
def exact_patch(
    model: PreTrainedModel,
    layers: nn.ModuleList,
    clean: Tensor,
    noise: Tensor,
    metric: Metric,
    candidates: list[tuple[int, int | None, int]],
    site: Site = Site.DOWN_IN,
    batch_size: int = 16,
    log: Callable[[str], None] | None = None,
    readout: Callable[[Tensor], Tensor] | None = None,
) -> tuple[Tensor, float]:
    """Metric change from patching each ``(layer, position or None for all, neuron)`` to its
    corrupted value, one candidate per batch row. Returns ``(effects [len(candidates)], clean metric)``.

    ``metric`` is applied to ``readout(ids)``, by default the last position's logits; see
    ``fp32_logit_readout`` for a finer-grained readout on a bf16 model."""
    site = Site(site)
    layer_set = sorted({c[0] for c in candidates})
    a_noise = read_site(model, layers, noise, site, layer_set)
    dev = model.device
    if readout is None:
        def readout(ids: Tensor) -> Tensor:
            return model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1]
    base = metric(readout(clean.to(dev))).item()
    effects = torch.empty(len(candidates))
    t0 = time.time()
    for s in range(0, len(candidates), batch_size):
        chunk = candidates[s : s + batch_size]
        per_layer: dict[int, list[tuple[int, int | None, int]]] = {}
        for r, (L, t, n) in enumerate(chunk):
            per_layer.setdefault(L, []).append((r, t, n))

        def patch(L: int):
            def fn(x: Tensor) -> Tensor:
                out = x.clone()
                src = a_noise[L][0].to(x.dtype)
                for r, t, n in per_layer[L]:
                    if t is None:
                        out[r, :, n] = src[:, n]
                    else:
                        out[r, t, n] = src[t, n]
                return out

            return fn

        hs = [hook_site(layers[L], site, patch(L)) for L in per_layer]
        try:
            ids = clean.to(dev).expand(len(chunk), -1)
            m = metric(readout(ids)).float().cpu()
        finally:
            for h in hs:
                h.remove()
        effects[s : s + len(chunk)] = m - base
        if log and (s // batch_size) % 20 == 0:
            log(f"patched {s + len(chunk)}/{len(candidates)} ({time.time() - t0:.0f}s)")
    return effects, base


def fp32_logit_readout(model: PreTrainedModel, tokens: list[int]) -> Callable[[Tensor], Tensor] | None:
    """Readout for ``exact_patch``: the logits of ``tokens`` at the last position, computed in
    float32 from the final (normed) hidden state, as ``[batch, len(tokens)]``. A bf16 model's own
    logits come in steps of 1/32 to 1/16 at typical values, coarser than most single-neuron patch
    effects. Returns None when the output weights aren't a float tensor (e.g. 4-bit)."""
    w = model.get_output_embeddings().weight
    if not w.is_floating_point() or w.dim() != 2:
        return None
    W = w[tokens].detach().float()
    _, path = find_layers(model)
    body = model
    for p in path.split(".")[:-1]:
        body = getattr(body, p)

    @torch.no_grad()
    def readout(ids: Tensor) -> Tensor:
        h = body(input_ids=ids, use_cache=False).last_hidden_state[:, -1].float()
        return h @ W.T

    return readout


def integrated_gradients(
    model: PreTrainedModel,
    layers: nn.ModuleList,
    ids: Tensor,
    metric: Metric,
    layer_ids: Iterable[int] | None = None,
    position: int = -1,
    steps: int = 20,
    site: Site = Site.DOWN_IN,
) -> Tensor:
    """Integrated gradients of ``metric`` w.r.t. each layer's activation at one position.

    For layer L with activation ``a`` at ``position``, the baseline is 0 and
    ``IG = a * mean_k dM/da (k/steps * a)`` for k = 1..steps (Riemann sum, as in Dai et al.'s
    code). All steps run as one batch; layers are attributed one at a time. Returns
    ``[n_layers_read, width]`` float32 on CPU.
    """
    check_layers(layers)
    site = Site(site)
    layer_ids = list(layer_ids) if layer_ids is not None else list(range(len(layers)))
    dev = model.device
    ids = ids.to(dev)
    T = ids.shape[1]
    pos = position % T
    clean = read_site(model, layers, ids, site, layer_ids)
    alphas = torch.arange(1, steps + 1, dtype=torch.float32, device=dev) / steps
    batch = ids.expand(steps, -1)
    out = torch.zeros(len(layer_ids), site_width(layers[layer_ids[0]], site))
    grads_were = {n: p.requires_grad for n, p in model.named_parameters()}
    model.requires_grad_(False)
    try:
        for j, L in enumerate(layer_ids):
            a = clean[L][0, pos].float()  # [W]
            scaled = (alphas[:, None] * a[None]).requires_grad_(True)

            def put(x: Tensor) -> Tensor:
                o = x.clone()
                o[:, pos] = scaled.to(x.dtype)
                return o

            h = hook_site(layers[L], site, put)
            try:
                with torch.enable_grad():
                    logits = model(input_ids=batch, use_cache=False, logits_to_keep=1).logits[:, -1]
                    (g,) = torch.autograd.grad(metric(logits).sum(), scaled)
            finally:
                h.remove()
            out[j] = (a * g.float().mean(0)).cpu()
    finally:
        for n, p in model.named_parameters():
            p.requires_grad_(grads_were[n])
    return out
