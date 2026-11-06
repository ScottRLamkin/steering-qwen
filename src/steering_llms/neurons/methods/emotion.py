"""Emotion neurons by activation-probability entropy (Emotion & Rhetoric neurons, arXiv 2604.17255),
evaluated as the paper does for classification: ablate one emotion's neurons and see whether the
model loses that emotion specifically.

Find: dair-ai/emotion texts (six labels) are read token by token. A neuron's firing rate
(``act > 0``) per label, normalized over labels, has an entropy; the lowest 1% are
emotion-specific and belong to the label where they fire most (with at least ``--min-rate``).

Change: zero one label's neurons at every position while the model classifies held-out texts
zero-shot (label read from the probabilities of the six label words). Reported: the accuracy drop
on that label vs on the others, and a random control. The paper's steering variants (masking,
adaptive damping, adding beta * mean) need hyperparameters it doesn't give.
"""

import argparse

import torch

from .. import data
from ..collect import Collector, text_sample
from ..evaluate import label_probs, random_like
from ..intervene import Edit, Intervention, edits_from_set
from ..neuronset import NeuronSet
from ..select import bottom_quantile_mask, to_set
from ..sites import Site
from ..stats import lape_entropy
from .common import Ctx

NAME = "emotion"
HELP = "Emotion neurons: lowest-entropy firing across labels; ablate and classify"

PROMPT = ("Which emotion does this text express: sadness, joy, love, anger, fear or surprise? "
          "Answer with one word.\n\nText: {text}")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-per", type=int, default=300, help="training texts per label")
    p.add_argument("--q", type=float, default=0.01)
    p.add_argument("--min-rate", type=float, default=0.05)
    p.add_argument("--n-eval", type=int, default=20, help="test texts per label")


def discover(ctx: Ctx, args: argparse.Namespace) -> dict[str, NeuronSet]:
    rows = data.emotion("train", seed=args.seed)
    by: dict[str, list[str]] = {}
    for t, lab in rows:
        if len(by.setdefault(lab, [])) < args.n_per:
            by[lab].append(t)
    samples = [text_sample(ctx.tok, t, lab) for lab, ts in by.items() for t in ts]
    layer_ids = list(range(ctx.n_layers))
    st = Collector(ctx.layers, Site.ACT, layer_ids).run(ctx.model, samples, args.batch_size * 2, log=ctx.log)
    labels = data.EMOTIONS
    groups = [st[lab] for lab in labels]
    h = lape_entropy(groups)
    rates = torch.stack([g.frac_pos for g in groups])  # [K, L, W]
    low = bottom_quantile_mask(h, args.q)
    owner = rates.argmax(0)
    top_rate = rates.amax(0)
    sets = {}
    for k, lab in enumerate(labels):
        m = low & (owner == k) & (top_rate >= args.min_rate)
        sets[lab] = to_set(m, -h, layer_ids, Site.ACT, extras={"rate": rates[k], "entropy": h}, label=lab,
                           sort_by_abs=False, meta={"method": NAME, "q": args.q})
        ctx.log(f"{lab}: {len(sets[lab])} neurons, layers {sets[lab].layer_counts()}")
    return sets


def edits(sets: dict[str, NeuronSet], label: str, direction: str = "-") -> list[Edit]:
    return edits_from_set(sets[label], "mul", 0.0 if direction == "-" else 2.0, scope="all")


def evaluate(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    labels = data.EMOTIONS
    test = data.emotion("test", seed=args.seed)
    ev: dict[str, list[str]] = {}
    for t, lab in test:
        if len(ev.setdefault(lab, [])) < args.n_eval:
            ev[lab].append(t)
    words = [w.capitalize() for w in labels]

    def accuracy(e: list[Edit]) -> dict[str, float]:
        acc = {}
        with Intervention(ctx.layers, e):
            for lab, texts in ev.items():
                ok = 0
                for t in texts:
                    p = label_probs(ctx.model, ctx.tok, PROMPT.format(text=t), words)
                    ok += max(p, key=p.get).lower() == lab
                acc[lab] = ok / len(texts)
        return acc

    base = accuracy([])
    ctx.log("base accuracy " + " ".join(f"{k} {v:.2f}" for k, v in base.items()))
    out = {"base": base, "ablate": {}}
    for lab in labels:
        if not len(sets[lab]):
            continue
        rows = {}
        for name, e in (("own", edits(sets, lab)), ("random", edits({lab: random_like(sets[lab], ctx.width, args.seed)}, lab))):
            acc = accuracy(e)
            self_drop = base[lab] - acc[lab]
            other = [base[x] - acc[x] for x in labels if x != lab and x in acc]
            rows[name] = {"acc": acc, "self_drop": self_drop, "other_drop": sum(other) / len(other)}
        out["ablate"][lab] = rows
        ctx.log(f"ablate {lab} ({len(sets[lab])}): self drop {rows['own']['self_drop']:+.2f} "
                f"others {rows['own']['other_drop']:+.2f}; random self {rows['random']['self_drop']:+.2f}")
    return out
