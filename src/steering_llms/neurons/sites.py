"""Where a neuron lives in a SwiGLU MLP, and hooks that read or edit it there.

Qwen3.5's MLP is ``down_proj(act_fn(gate_proj(x)) * up_proj(x))``. Papers read "neurons" at
different points of it, so every neuron set and every edit names its :class:`Site`:

* ``gate``      ``gate_proj(x)``, before the nonlinearity (Psychological Concept Neurons, AurA)
* ``act``       ``silu(gate_proj(x))``; "firing" means > 0 (NPTI, IRiS, sNeuron-TST, emotion)
* ``up``        ``up_proj(x)`` (AurA)
* ``down_in``   ``silu(gate) * up``, the input to ``down_proj`` (HedwigAI, DPN-LE, gender, ...)
* ``down_out``  ``down_proj``'s output, d_model wide (AurA; what :class:`Steerer` adds to)

All hooks work for plain bf16 ``nn.Linear`` and for bitsandbytes ``Linear4bit``.
"""

from collections.abc import Callable
from enum import Enum

from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


class Site(str, Enum):
    GATE = "gate"
    ACT = "act"
    UP = "up"
    DOWN_IN = "down_in"
    DOWN_OUT = "down_out"


def site_module(layer: nn.Module, site: Site) -> tuple[nn.Module, bool]:
    """The module to hook for ``site`` and whether the site is that module's *input*."""
    mlp = layer.mlp
    site = Site(site)
    if site is Site.GATE:
        return mlp.gate_proj, False
    if site is Site.ACT:
        return mlp.act_fn, False
    if site is Site.UP:
        return mlp.up_proj, False
    if site is Site.DOWN_IN:
        return mlp.down_proj, True
    return mlp.down_proj, False


def site_width(layer: nn.Module, site: Site) -> int:
    dp = layer.mlp.down_proj
    return dp.out_features if Site(site) is Site.DOWN_OUT else dp.in_features


def check_layers(layers: nn.ModuleList) -> None:
    """Per-layer ``act`` hooks need a separate activation module in every layer."""
    acts = [id(layer.mlp.act_fn) for layer in layers]
    if len(set(acts)) != len(acts):
        raise RuntimeError("MLP layers share one act_fn module; per-layer 'act' hooks would fire everywhere")


SiteFn = Callable[[Tensor], Tensor | None]


def hook_site(layer: nn.Module, site: Site, fn: SiteFn) -> RemovableHandle:
    """Call ``fn(x)`` with the site's tensor ``[batch, seq, width]``; a returned tensor replaces it."""
    module, is_input = site_module(layer, site)
    if is_input:

        def pre(mod, args):
            out = fn(args[0])
            return None if out is None else (out, *args[1:])

        return module.register_forward_pre_hook(pre)

    def post(mod, args, output):
        return fn(output)

    return module.register_forward_hook(post)
