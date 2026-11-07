"""Psychological Concept Neurons (arXiv 2604.11802): Big Five neurons by average precision.

Find: each of PersonalityBench's 300 IPIP facet statements (60 per domain) is shown to the model
as an inventory item. At the last prompt token, a neuron's pre-SiLU ``gate`` value scores every
item; its average precision at ranking one domain's items above the other four domains' items is
its concept score. The ``k`` best neurons per domain are kept.

Change: set those neurons to their 99th percentile over all items (boost) or their 1st
percentile (suppress), at the last prompt position only, which produces the first answer token.

The paper uses BFI-2's 60 items, which are free for research but not redistributable; these
IPIP statements are public domain.
"""

import argparse

import torch

from ..collect import Collector, prompt_sample
from ..intervene import Edit, edits_from_set
from ..neuronset import NeuronSet
from .. import select
from ..evaluate import random_like
from ..select import to_set, top_k_mask
from ..sites import Site
from ..stats import average_precision, quantile_cols
from . import personality
from .common import Ctx

NAME = "pcn"
HELP = "Psychological Concept Neurons: average precision of gate values over inventory items"

ITEM = 'Here is a statement someone might make about themselves: "I {item}"'


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--k", type=int, default=50, help="neurons per domain")
    p.add_argument("--hi-q", type=float, default=0.99)
    p.add_argument("--lo-q", type=float, default=0.01)
    p.add_argument("--pcn-scope", default="prefill_last", help="position scope of the edit (paper: prefill_last)")


def discover(ctx: Ctx, args: argparse.Namespace) -> dict[str, NeuronSet]:
    items = personality.facet_items()
    samples = [prompt_sample(ctx.tok, ITEM.format(item=personality.lower_first(s)), group="items") for _, s, _ in items]
    layer_ids = list(range(ctx.n_layers))
    st = Collector(ctx.layers, Site.GATE, layer_ids, pool="last", moments=False).run(
        ctx.model, samples, args.batch_size, log=ctx.log
    )["items"]
    X = st.sample_matrix()  # [items, L, W]
    hi = quantile_cols(X, args.hi_q)
    lo = quantile_cols(X, args.lo_q)
    domains = [d for d, _, _ in items]
    sets = {}
    for trait in args.traits:
        y = torch.tensor([d == trait for d in domains])
        ap = average_precision(X, y)
        mask = top_k_mask(ap, args.k, by_abs=False)
        sets[trait] = to_set(
            mask, ap, layer_ids, Site.GATE, extras={"p_hi": hi, "p_lo": lo, "mean": X.mean(0)},
            label=trait, meta={"method": NAME, "n_items": len(items), "k": args.k, "chance_ap": float(y.float().mean())},
            sort_by_abs=False,
        )
        # control: random neurons in the same layers, set to *their own* percentiles
        rmask = select.to_mask(random_like(sets[trait], ctx.width, args.seed), layer_ids, ctx.width)
        sets[f"{trait}@random"] = to_set(
            rmask, ap, layer_ids, Site.GATE, extras={"p_hi": hi, "p_lo": lo, "mean": X.mean(0)},
            label=f"{trait} random", meta={"method": NAME, "control": True}, sort_by_abs=False,
        )
        top = sets[trait]
        ctx.log(f"{trait}: AP {min(top.score):.3f}..{max(top.score):.3f} (chance {y.float().mean():.2f}), layers {top.layer_counts()}")
    return sets


def edits(sets: dict[str, NeuronSet], trait: str, direction: str, scope: str = "prefill_last") -> list[Edit]:
    return edits_from_set(sets[trait], "set", "p_hi" if direction == "+" else "p_lo", scope=scope)


def evaluate(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    fn = lambda s, t, d: edits(s, t, d, scope=args.pcn_scope)  # noqa: E731
    main = {k: v for k, v in sets.items() if "@" not in k}
    rnd = {k.split("@")[0]: v for k, v in sets.items() if k.endswith("@random")}
    return personality.evaluate(ctx, main, fn, args.traits, n_questions=args.eval_questions, seed=args.seed,
                                with_replies=not args.no_replies, random_sets=rnd or None)


def score(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    from . import persona_score

    fn = lambda s, t, d: edits(s, t, d, scope=args.pcn_scope)  # noqa: E731
    main = {k: v for k, v in sets.items() if "@" not in k}
    rnd = {k.split("@")[0]: v for k, v in sets.items() if k.endswith("@random")}
    conds = personality.conditions(ctx, main, fn, args.traits, args.seed, random_sets=rnd or None)
    return persona_score.score(ctx, conds, args.traits)
