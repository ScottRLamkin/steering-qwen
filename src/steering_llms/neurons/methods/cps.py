"""Contrastive Pair Search (arXiv 2605.12290): the neurons that separate harmful from matched
benign requests, found from last-token activations alone.

Find: JailbreakBench's 100 harmful behaviors and their 100 topic-matched benign twins (half
of each for discovery, half held out). At the last prompt token, ``down_in``'s mean difference
harmful - benign; the top 0.1% by ``|delta|`` (~295 neurons). "Universal" neurons, among the
top 0.1% by ``|activation|`` on at least 80% of a set of ordinary prompts, are removed: they are
large on everything.

Change: multiply the neurons by ``m`` at every position: ``m = 0`` ablates them (the paper's
refusal-removal experiment), ``m = 2`` amplifies them.

Measured by the refusal-phrase rate on the held-out harmful and benign requests. Only rates and
refusal flags are saved, not the replies.
"""

import argparse

import torch

from ... import discovery
from .. import data
from ..collect import Collector, prompt_sample
from ..evaluate import random_like, refusal_rate
from ..gen import generate_batch
from ..intervene import Edit, Intervention, edits_from_set
from ..neuronset import NeuronSet
from ..select import frequent, to_set, top_k_mask
from ..sites import Site
from ..stats import mean_diff
from .common import Ctx

NAME = "cps"
HELP = "Contrastive Pair Search: harmful-vs-benign last-token mean difference; scale by m"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--frac", type=float, default=0.001, help="fraction of all neurons kept")
    p.add_argument("--universal", type=float, default=0.8)
    p.add_argument("--max-new-tokens", type=int, default=48)


def split(kind: str) -> tuple[list[str], list[str]]:
    rows = sorted(data.jbb(kind), key=lambda r: int(r["Index"]))
    goals = [r["Goal"] for r in rows]
    return goals[0::2], goals[1::2]


def diverse_prompts() -> list[str]:
    return discovery.QUESTIONS + discovery.HELDOUT + [f"Summarize this review: {s}" for s, _ in data.sst2("validation", n=32)]


def discover(ctx: Ctx, args: argparse.Namespace) -> dict[str, NeuronSet]:
    harm, _ = split("harmful")
    benign, _ = split("benign")
    layer_ids = list(range(ctx.n_layers))
    samples = [prompt_sample(ctx.tok, g, "harmful") for g in harm] + [prompt_sample(ctx.tok, g, "benign") for g in benign]
    st = Collector(ctx.layers, Site.DOWN_IN, layer_ids).run(ctx.model, samples, args.batch_size, log=ctx.log)
    delta = mean_diff(st["harmful"], st["benign"])
    k = int(args.frac * ctx.n_layers * ctx.width)
    mask = top_k_mask(delta, k)
    div = Collector(ctx.layers, Site.DOWN_IN, layer_ids, pool="last", moments=False).run(
        ctx.model, [prompt_sample(ctx.tok, p, "d") for p in diverse_prompts()], args.batch_size
    )["d"].sample_matrix()
    universal = frequent([top_k_mask(x, k) for x in div], args.universal)
    removed = int((mask & universal).sum())
    mask &= ~universal
    ns = to_set(mask, delta, layer_ids, Site.DOWN_IN, label="refusal",
                meta={"method": NAME, "k": k, "universal_removed": removed, "n_universal": int(universal.sum())})
    ctx.log(f"refusal: {len(ns)} neurons ({removed} universal removed), layers {ns.layer_counts()}")
    return {"refusal": ns}


def edits(sets: dict[str, NeuronSet], key: str = "refusal", direction: str = "-", m: float | None = None) -> list[Edit]:
    if m is None:
        m = 0.0 if direction == "-" else 2.0
    return edits_from_set(sets[key], "mul", m, scope="all")


def evaluate(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    _, harm = split("harmful")
    _, benign = split("benign")
    rnd = {"refusal": random_like(sets["refusal"], ctx.width, args.seed)}
    conds = {"base": [], "m=0": edits(sets, m=0.0), "m=2": edits(sets, m=2.0), "random m=0": edits(rnd, m=0.0)}
    out = {"n_neurons": len(sets["refusal"]), "conditions": {}}
    for name, e in conds.items():
        r = {}
        with Intervention(ctx.layers, e):
            for kind, prompts in (("harmful", harm), ("benign", benign)):
                texts = generate_batch(ctx.model, ctx.tok, prompts, max_new_tokens=args.max_new_tokens,
                                       batch_size=args.batch_size)
                r[f"refusal_{kind}"] = refusal_rate(texts)
                r[f"flags_{kind}"] = [refusal_rate([t]) for t in texts]
        out["conditions"][name] = r
        ctx.log(f"{name}: refusal harmful {r['refusal_harmful']:.2f} benign {r['refusal_benign']:.2f}")
    return out


__all__ = ["torch"]
