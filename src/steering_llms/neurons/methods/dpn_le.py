"""DPN-LE (arXiv 2604.27929): personality neurons from prompt-only activation contrasts.

Find: for each trait, PersonalityBench search questions are asked under a high-pole and a
low-pole persona (DPN-LE's prompt). At the last prompt token, ``down_in`` activations give the
mean difference ``s = mean_high - mean_low`` and Cohen's ``d``. In each layer of the model's
second half, keep neurons with ``|s|`` above that layer's 99.5th percentile and ``|d| > tau``
(0.3, the paper's Qwen setting).

Change: ``h + direction * gamma * w * s`` on generated tokens, with weights ``w`` from 0.75 to 1.0
by the neuron's ``|d|`` rank within its layer.

We ask every question under both poles (paired) instead of one random pole per question.
"""

import argparse
import random

import torch

from .. import data
from ..collect import Collector, prompt_sample
from ..intervene import Edit, edits_from_set
from ..neuronset import NeuronSet
from ..select import quantile_mask, threshold_mask, to_set
from ..sites import Site
from ..stats import cohens_d, mean_diff
from . import personality
from .common import Ctx, set_summary

NAME = "dpn-le"
HELP = "DPN-LE: prompt-only mean-difference + Cohen's d personality neurons (Big Five)"

TEMPLATE = (
    "You will find a personality description followed by a question below. I want you to fully immerse "
    "yourself in the persona described.\n###Personality description: {description}\n###Question: {question}\n###Response:"
)


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-questions", type=int, default=200, help="search questions per trait (paper: 1000)")
    p.add_argument("--quantile", type=float, default=0.995)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=1.0)


def rank_weights(score: torch.Tensor, mask: torch.Tensor, lo: float = 0.75, hi: float = 1.0) -> torch.Tensor:
    """Per layer, the selected neurons get weights ``lo..hi`` by rank of ``|score|`` (highest = hi)."""
    w = torch.zeros_like(score, dtype=torch.float32)
    for j in range(score.shape[0]):
        idx = mask[j].nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        order = score[j, idx].abs().argsort()
        w[j, idx[order]] = torch.linspace(lo, hi, len(idx)) if len(idx) > 1 else torch.tensor([hi])
    return w


def discover(ctx: Ctx, args: argparse.Namespace) -> dict[str, NeuronSet]:
    layer_ids = list(range(ctx.n_layers // 2, ctx.n_layers))
    desc = data.personality_descriptions()
    sets = {}
    for trait in args.traits:
        rng = random.Random(args.seed)
        qs = data.personality_questions(trait, "search", n=args.n_questions, seed=args.seed)
        samples = []
        for r in qs:
            for pole in "+-":
                text = TEMPLATE.format(description=rng.choice(desc[trait][pole]), question=r["question"])
                samples.append(prompt_sample(ctx.tok, text, group=pole))
        st = Collector(ctx.layers, Site.DOWN_IN, layer_ids).run(ctx.model, samples, args.batch_size, log=ctx.log)
        s = mean_diff(st["+"], st["-"])
        d = cohens_d(st["+"], st["-"])
        mask = quantile_mask(s, args.quantile, per_layer=True) & threshold_mask(d, args.tau)
        w = rank_weights(d, mask)
        sets[trait] = to_set(
            mask, d, layer_ids, Site.DOWN_IN,
            extras={"s": s, "weight": w, "mean_hi": st["+"].mean, "mean_lo": st["-"].mean},
            label=trait,
            meta={"method": NAME, "n_questions": len(qs), "quantile": args.quantile, "tau": args.tau, "layers": layer_ids},
        )
        ctx.log(f"{trait}: {len(sets[trait])} neurons, per layer {sets[trait].layer_counts()}")
    return sets


def edits(sets: dict[str, NeuronSet], trait: str, direction: str, gamma: float = 1.0, scope: str = "decode") -> list[Edit]:
    sign = 1.0 if direction == "+" else -1.0
    return edits_from_set(sets[trait], "add", lambda r: sign * gamma * r["weight"] * r["s"], scope=scope)


def evaluate(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    fn = lambda s, t, d: edits(s, t, d, gamma=args.gamma)  # noqa: E731
    return personality.evaluate(ctx, sets, fn, args.traits, n_questions=args.eval_questions, seed=args.seed,
                                with_replies=not args.no_replies)


def summary(sets: dict[str, NeuronSet]) -> dict:
    return set_summary(sets)
