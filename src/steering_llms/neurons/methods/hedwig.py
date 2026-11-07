"""HedwigAI's single-neuron personalities (the gist, reproduced on branch neuron-discovery), as
neuron sets, so they can be compared with the paper recipes.

Find: nothing new. Reads the top screened neurons per trait from a finished
``scripts/find_neurons.py`` run (``screen.json``, ``medians.json``).

Change: HedwigAI adds ``dose * median_norm(L) * col / ||col||`` to the residual, where ``col`` is
the neuron's ``down_proj`` column. ``down_proj`` has no bias, so that is exactly adding
``dose * median_norm(L) / ||col||`` to the neuron's ``down_in`` value, at every position. The
evaluation checks this equivalence against :class:`Steerer` on the real model.

Evaluate: each trait's top neuron, pushed at its signed dose, on the Big Five questionnaire, which
shows where HedwigAI's nine personalities land on the Big Five; plus the equivalence check.
"""

import argparse
import json
from pathlib import Path

import torch

from ...steer import Steerer
from ..intervene import Edit, Intervention
from ..neuronset import NeuronSet
from ..sites import Site
from . import personality
from .common import Ctx

NAME = "hedwig"
HELP = "HedwigAI's screened single neurons (from a find_neurons.py run) as neuron sets"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--hedwig-run", type=Path, default=Path("runs/torch-bf16"))
    p.add_argument("--top", type=int, default=3, help="neurons kept per trait")
    p.add_argument("--dose", type=float, default=1.0, help="multiplies each neuron's signed screening dose")
    p.add_argument("--fluent-only", action="store_true",
                   help="skip neurons the search's rate stage flagged as broken (ratings.json); unrated ones are kept, in rank order")


def discover(ctx: Ctx, args: argparse.Namespace) -> dict[str, NeuronSet]:
    screen = json.loads((args.hedwig_run / "screen.json").read_text())
    med = {int(k): float(v) for k, v in json.loads((args.hedwig_run / "medians.json").read_text()).items()}
    st = Steerer(ctx.model, ctx.tok)
    sets = {}
    broken: set[str] = set()
    if getattr(args, "fluent_only", False):
        ratings = json.loads((args.hedwig_run / "ratings.json").read_text())["conditions"]
        broken = {k for k, r in ratings.items() if r["broken"]}
        ctx.log(f"fluent only: skipping {len(broken)} broken neurons: {', '.join(sorted(broken))}")
    for trait, rows in screen["top"].items():
        if broken:
            rows = [r for r in rows if f"L{r['layer']}N{r['neuron']}{'+' if r['dose'] > 0 else '-'}" not in broken]
        rows = rows[: args.top]
        layer = [r["layer"] for r in rows]
        index = [r["neuron"] for r in rows]
        norms = [st.down_col(L, n).norm().item() for L, n in zip(layer, index)]
        doses = [r["dose"] for r in rows]
        sets[trait] = NeuronSet(
            Site.DOWN_IN, layer, index, [r["specificity"] for r in rows],
            {
                "dose": doses,
                "median_norm": [med[L] for L in layer],
                "col_norm": norms,
                "value": [d * med[L] / c for d, L, c in zip(doses, layer, norms)],
            },
            trait,
            {"method": NAME, "source": str(args.hedwig_run)},
        )
        ctx.log(f"{trait}: " + ", ".join(f"L{L}N{n}{'+' if d > 0 else '-'}" for L, n, d in zip(layer, index, doses)))
    return sets


def edits(sets: dict[str, NeuronSet], trait: str, direction: str = "+", k: int = 1, dose: float = 1.0) -> list[Edit]:
    """The top ``k`` neurons of ``trait`` pushed at ``dose`` times their signed screening dose
    (``direction`` ``-`` flips the push)."""
    ns = sets[trait]
    sign = dose if direction == "+" else -dose
    return [Edit(ns.layer[i], [ns.index[i]], "add", sign * ns.extras["value"][i], Site.DOWN_IN, "all") for i in range(min(k, len(ns)))]


@torch.no_grad()
def equivalence(ctx: Ctx, L: int, n: int, dose: float, value: float) -> dict:
    """Max |logit difference| between Steerer's residual push and the down_in edit."""
    st = Steerer(ctx.model, ctx.tok)
    st.use_medians(json.loads((Path(ctx.run_dir) / "medians_used.json").read_text()))
    ids = torch.tensor([ctx.tok.encode("What do you think about the ocean at night?")], device=ctx.model.device)
    with st.steering([(L, n, dose)], mode="dose"):
        a = ctx.model(input_ids=ids, use_cache=False).logits.float()
    with Intervention(ctx.layers, [Edit(L, [n], "add", value, Site.DOWN_IN)]):
        b = ctx.model(input_ids=ids, use_cache=False).logits.float()
    base = ctx.model(input_ids=ids, use_cache=False).logits.float()
    return {"max_abs_diff": (a - b).abs().max().item(), "max_abs_effect": (a - base).abs().max().item()}


def evaluate(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    med = json.loads((args.hedwig_run / "medians.json").read_text())
    ctx.save_json("medians_used.json", med)
    first = next(iter(sets.values()))
    eq = equivalence(ctx, first.layer[0], first.index[0], first.extras["dose"][0], first.extras["value"][0])
    ctx.log(f"Steerer vs down_in edit: {eq}")
    base = {d: personality.questionnaire(ctx, d, None) for d in args.traits}
    out = {"equivalence": eq, "base": base, "traits": {}}
    for trait, ns in sets.items():
        e = edits(sets, trait, "+", k=1, dose=args.dose)
        ctx.save_edits(f"edits/{trait}.json", e)
        q = {d: personality.questionnaire(ctx, d, e) for d in args.traits}
        out["traits"][trait] = {
            "neuron": f"L{ns.layer[0]}N{ns.index[0]}{'+' if ns.extras['dose'][0] > 0 else '-'}",
            "questionnaire": q,
            "shift": {d: q[d] - base[d] for d in q},
        }
        ctx.log(f"{trait} {out['traits'][trait]['neuron']}: " + " ".join(f"{d[:4]} {v:+.2f}" for d, v in out["traits"][trait]["shift"].items()))
    return out


def score(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    """Which Big Five pole each HedwigAI neuron pushes toward, by the persona log-prob contrast."""
    from ..evaluate import random_like
    from . import persona_score

    conds = {}
    for t in args.traits:
        conds[t] = {name: edits(sets, name, "+", k=1, dose=args.dose) for name in sets}
        for name in sets:
            conds[t][f"{name} random"] = edits({name: random_like(sets[name].top(1), ctx.width, args.seed)}, name, "+", k=1, dose=args.dose)
    return persona_score.score(ctx, conds, args.traits)
