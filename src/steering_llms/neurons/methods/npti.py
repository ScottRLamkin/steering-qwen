"""NPTI, Neuron-based Personality Trait Induction (arXiv 2410.12327), following its released code.

Find: for each trait and pole, the model answers PersonalityBench search questions while role-
playing a random one of NPTI's 80 personas for that pole (greedy, repetition penalty 1.1). The
answers are re-read and every answer token's ``act = silu(gate)`` is counted. A neuron's
``delta = Pr_pole(act > 0) - Pr_other(act > 0)``. Sorted by ``delta``, the top ``min_required``
(2000) neurons are kept, plus any further ones with ``delta >= 0.1``. ``a95`` is each neuron's
95th-percentile activation on its pole (histogram of 0.01-wide bins over 0..3).

Change (on generated tokens only): target-pole neurons get
``act += gamma * a95 * sigmoid(10 * (delta - 0.15))``, with a random 90% of them each step;
opposite-pole neurons are clamped to ``min(act, 0)`` in layers that also hold target neurons.
"""

import argparse
import random

import torch

from .. import data
from ..collect import Collector, answer_sample
from ..gen import generate_batch
from ..intervene import Edit
from ..neuronset import NeuronSet
from ..select import to_set
from ..sites import Site
from . import personality
from .common import Ctx

NAME = "npti"
HELP = "NPTI: firing-rate difference on persona answers; add a95 on generated tokens"

# search_neuron.py's prompt, verbatim
TEMPLATE = """
You will find a personality description followed by a question below. I want you to forget who you are and fully immerse yourself in the persona described, adopting not only their perspective but also their tone and attitude. With this new identity in mind, please respond to the question.
Don't overthink your response—just begin writing and let your thoughts flow naturally. Spelling and grammar are not important here; what's essential is capturing the essence of this personality in your answer. Try to keep your response under 300 words.
###Personality description:{personality}

###Question:{question}

###Response:
"""


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-questions", type=int, default=24, help="search questions per pole (paper: all 36k)")
    p.add_argument("--max-new-tokens", type=int, default=160, help="answer length for the search (paper: 500)")
    p.add_argument("--min-required", type=int, default=2000)
    p.add_argument("--min-diff", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=1.4, help="NPTI's val; the code sweeps 0.6..1.5")
    p.add_argument("--keep-prob", type=float, default=0.9)


def select_pole(delta: torch.Tensor, min_required: int, min_diff: float) -> torch.Tensor:
    flat = delta.reshape(-1)
    n = max(min_required, int((flat >= min_diff).sum()))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[flat.argsort(descending=True)[:n]] = True
    return mask.view_as(delta)


def answers_for(ctx: Ctx, trait: str, pole: str, args: argparse.Namespace) -> list[dict]:
    def make():
        desc = data.personality_descriptions()[trait][pole]
        rng = random.Random(f"{args.seed}{trait}{pole}")
        qs = data.personality_questions(trait, "search", n=args.n_questions, seed=args.seed)
        prompts = [TEMPLATE.format(personality=rng.choice(desc), question=r["question"]) for r in qs]
        texts = generate_batch(ctx.model, ctx.tok, prompts, max_new_tokens=args.max_new_tokens,
                               repetition_penalty=1.1, batch_size=args.batch_size, log=ctx.log)
        return [{"prompt": p, "answer": a} for p, a in zip(prompts, texts)]

    return ctx.cached_json(f"answers/{trait}{pole}.json", make)


def discover(ctx: Ctx, args: argparse.Namespace) -> dict[str, NeuronSet]:
    layer_ids = list(range(ctx.n_layers))
    sets = {}
    for trait in args.traits:
        samples = []
        for pole in "+-":
            for r in answers_for(ctx, trait, pole, args):
                samples.append(answer_sample(ctx.tok, r["prompt"], r["answer"], group=pole))
        st = Collector(ctx.layers, Site.ACT, layer_ids, hist=(0.0, 3.0, 300)).run(
            ctx.model, samples, max(1, args.batch_size // 2), log=ctx.log
        )
        for pole, other in (("+", "-"), ("-", "+")):
            delta = st[pole].frac_pos - st[other].frac_pos
            mask = select_pole(delta, args.min_required, args.min_diff)
            a95 = st[pole].percentile(95).clamp_min(0)
            key = f"{trait}{pole}"
            sets[key] = to_set(
                mask, delta, layer_ids, Site.ACT, extras={"a95": a95, "rate": st[pole].frac_pos},
                label=key, sort_by_abs=False,
                meta={"method": NAME, "n_questions": args.n_questions, "tokens": st[pole].n_tokens,
                      "min_required": args.min_required, "min_diff": args.min_diff},
            )
            ns = sets[key]
            ctx.log(f"{key}: {len(ns)} neurons, delta {min(ns.score):.3f}..{max(ns.score):.3f}, "
                    f"{sum(1 for x in ns.score if x >= args.min_diff)} with delta >= {args.min_diff}")
        del st
    return sets


def edits(
    sets: dict[str, NeuronSet], trait: str, direction: str, gamma: float = 1.4, keep_prob: float = 0.9,
    scope: str = "decode",
) -> list[Edit]:
    other = "-" if direction == "+" else "+"
    target, opp = sets[f"{trait}{direction}"], sets[f"{trait}{other}"]
    t_layers = target.by_layer()
    o_layers = opp.by_layer()
    out = []
    for L, d in t_layers.items():
        if L in o_layers:  # deactivate first, as NPTI's forward does
            out.append(Edit(L, o_layers[L]["index"].tolist(), "clamp_max", 0.0, Site.ACT, scope))
        v = gamma * d["a95"] * torch.sigmoid(10 * (d["score"] - 0.15))
        out.append(Edit(L, d["index"].tolist(), "add", v.tolist(), Site.ACT, scope, keep_prob=keep_prob))
    return out


def evaluate(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    fn = lambda s, t, d: edits(s, t, d, gamma=args.gamma, keep_prob=args.keep_prob)  # noqa: E731
    return personality.evaluate(ctx, sets, fn, args.traits, n_questions=args.eval_questions, seed=args.seed,
                                with_replies=not args.no_replies)
