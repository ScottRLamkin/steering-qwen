"""AurA, "Whispering Experts" (arXiv 2407.12824): damp toxicity experts in proportion to how well
they detect toxicity.

Find: every MLP unit at three sites (``gate`` and ``up`` outputs, 9216 each, and ``down_proj``'s
output, 2560; 671,744 units in all) is max-pooled over each sentence's tokens. Its AUROC at telling
toxic from non-toxic sentences makes it an expert when > 0.5.

Change: multiply each expert's output by ``alpha = 1 - Gini = 2 * (1 - AUROC)`` at every position;
no hyperparameter. The strongest detectors are damped most, ones near chance barely at all.

Data: ParaDetox's toxic and neutral paraphrases (the paper uses Jigsaw, a manual Kaggle download).
Evaluation: toxicity of continuations of held-out toxic text prefixes (the paper uses
RealToxicityPrompts) and perplexity of neutral text under the intervention.
"""

import argparse

import torch

from .. import data
from ..collect import Collector, text_sample
from ..evaluate import Classifier, random_like
from ..gen import generate_batch
from ..intervene import Edit, Intervention, edits_from_set
from ..neuronset import NeuronSet
from ..select import threshold_mask, to_set
from ..sites import Site
from ..stats import auroc
from . import style
from .common import Ctx

NAME = "aura"
HELP = "AurA: AUROC experts at gate/up/down_out, damped by 1 - Gini at every position"
SITES = (Site.GATE, Site.UP, Site.DOWN_OUT)


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-texts", type=int, default=500, help="toxic and neutral sentences each")
    p.add_argument("--n-eval", type=int, default=24)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--min-auroc", type=float, default=0.5)


def discover(ctx: Ctx, args: argparse.Namespace) -> dict[str, NeuronSet]:
    pairs = data.paradetox(seed=args.seed)[: args.n_texts]
    samples = [text_sample(ctx.tok, a, "toxic") for a, _ in pairs] + [text_sample(ctx.tok, b, "neutral") for _, b in pairs]
    layer_ids = list(range(ctx.n_layers))
    sets = {}
    for site in SITES:
        st = Collector(ctx.layers, site, layer_ids, pool="max", moments=False).run(
            ctx.model, samples, args.batch_size * 2, log=ctx.log
        )
        au = auroc(st["toxic"].sample_matrix(), st["neutral"].sample_matrix())
        del st
        mask = threshold_mask(au, args.min_auroc, by_abs=False)
        alpha = (2 * (1 - au)).clamp(0, 1)
        sets[site.value] = to_set(mask, au, layer_ids, site, extras={"alpha": alpha}, label=f"toxic@{site.value}",
                                  sort_by_abs=False, meta={"method": NAME, "n_texts": args.n_texts})
        top = sorted(sets[site.value].score, reverse=True)[:5]
        ctx.log(f"{site.value}: {len(sets[site.value])} experts, top AUROC {[round(x, 3) for x in top]}")
    return sets


def edits(sets: dict[str, NeuronSet], key: str = "all", direction: str = "-") -> list[Edit]:
    keys = [s.value for s in SITES] if key == "all" else [key]
    return [e for k in keys if k in sets for e in edits_from_set(sets[k], "mul", "alpha", scope="all")]


def evaluate(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    pairs = data.paradetox(seed=args.seed)
    held = pairs[args.n_texts : args.n_texts + args.n_eval]
    prefixes = [" ".join(a.split()[: max(3, len(a.split()) // 2)]) for a, _ in held]
    neutral = [b for _, b in pairs[args.n_texts + 200 : args.n_texts + 260]]
    width = {k: ctx.layers[0].mlp.down_proj.out_features if k == "down_out" else ctx.width for k in sets}
    rnd = {k: random_like(v, width[k], args.seed) for k, v in sets.items()}
    conds = {"base": [], "aura": edits(sets), "random": edits(rnd)}
    for k in sets:
        conds[f"aura@{k}"] = edits(sets, k)
    clf = Classifier("toxicity", ctx.model.device)
    out = {"n_experts": {k: len(v) for k, v in sets.items()}, "conditions": {}}
    for name, e in conds.items():
        with Intervention(ctx.layers, e):
            conts = generate_batch(ctx.model, ctx.tok, prefixes, max_new_tokens=args.max_new_tokens,
                                   chat=False, batch_size=args.batch_size)
            ppl = style.text_ppl(ctx.model, ctx.tok, neutral)
        tox = clf(conts)
        r = {
            "toxicity": sum(p["toxic"] for p in tox) / len(tox),
            "toxic_frac": sum(p["toxic"] > 0.5 for p in tox) / len(tox),
            "neutral_ppl": ppl,
            "examples": [{"prefix": p, "continuation": c} for p, c in zip(prefixes[:6], conts[:6])],
        }
        out["conditions"][name] = r
        ctx.log(f"{name}: toxicity {r['toxicity']:.3f} ({r['toxic_frac']:.2f} > 0.5), neutral ppl {ppl:.2f}")
    return out


__all__ = ["torch"]
