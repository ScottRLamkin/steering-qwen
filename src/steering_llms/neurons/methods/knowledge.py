"""Knowledge Neurons (Dai et al., arXiv 2104.08696), for a left-to-right model.

Find: for each ParaRel fact (several paraphrased prompts that end right before the answer), the
attribution of every ``down_in`` neuron at the last prompt token to the probability of the
answer's first token is computed by integrated gradients (20 steps, baseline 0), one layer at a
time. Per prompt, neurons above ``0.2 *`` the prompt's largest attribution are kept; a fact's
knowledge neurons are those kept for more than ``p = 70%`` of its prompts, with ``p`` moved in
steps of 0.05 until 2-5 neurons remain (the paper's adjustment).

Change: zero the neurons (suppress) or double them (amplify) at the prediction position, and
measure the answer probability over the fact's prompts, plus the same edit's effect on facts of
other relations (should be small).
"""

import argparse
import random

import torch

from .. import data
from ..attribution import integrated_gradients, prob_metric
from ..intervene import Edit, Intervention, edits_from_set
from ..neuronset import NeuronSet
from ..select import consensus
from ..sites import Site
from .common import Ctx

NAME = "knowledge"
HELP = "Knowledge Neurons: integrated gradients on ParaRel facts; zero or double"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--facts-per-relation", type=int, default=2)
    p.add_argument("--max-prompts", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--layer-start", type=int, default=0)
    p.add_argument("--min-prob", type=float, default=0.02, help="skip facts the model doesn't know")
    p.add_argument("--p", type=float, default=0.7)


def answer_token(tok, answer: str) -> int:
    return tok.encode(" " + answer)[0]


@torch.no_grad()
def answer_prob(ctx: Ctx, prompt: str, token: int) -> float:
    ids = torch.tensor([ctx.tok.encode(prompt)], device=ctx.model.device)
    return ctx.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float().softmax(-1)[token].item()


def facts_known(ctx: Ctx, args: argparse.Namespace) -> list[dict]:
    def make():
        out = []
        for f in data.pararel(facts_per_relation=args.facts_per_relation, seed=args.seed):
            t = answer_token(ctx.tok, f["answer"])
            prompts = f["prompts"][: args.max_prompts]
            probs = [answer_prob(ctx, p, t) for p in prompts]
            if sum(probs) / len(probs) >= args.min_prob:
                out.append({**f, "prompts": prompts, "token": t, "base_probs": probs})
        return out

    return ctx.cached_json("facts.json", make)


def fact_neurons(attrs: list[torch.Tensor], layer_ids: list[int], p0: float) -> tuple[list[tuple[int, int]], float]:
    per_prompt = []
    for a in attrs:
        m = a > 0.2 * a.max()
        rows, cols = m.nonzero(as_tuple=True)
        per_prompt.append(NeuronSet(Site.DOWN_IN, [layer_ids[r] for r in rows.tolist()], cols.tolist(), [0.0] * len(rows)))
    p = p0
    keys = consensus(per_prompt, p)
    for _ in range(12):
        if len(keys) < 2 and p > 0.05:
            p -= 0.05
        elif len(keys) > 5 and p < 0.95:
            p += 0.05
        else:
            break
        keys = consensus(per_prompt, p)
    return keys, p


def discover(ctx: Ctx, args: argparse.Namespace) -> dict[str, NeuronSet]:
    facts = facts_known(ctx, args)
    layer_ids = list(range(args.layer_start, ctx.n_layers))
    sets = {}
    for i, f in enumerate(facts):
        attrs = []
        for p in f["prompts"]:
            ids = torch.tensor([ctx.tok.encode(p)])
            attrs.append(integrated_gradients(ctx.model, ctx.layers, ids, prob_metric(f["token"]), layer_ids, steps=args.steps))
        keys, p_used = fact_neurons(attrs, layer_ids, args.p)
        mean_attr = torch.stack(attrs).mean(0)
        key = f"{f['relation']}:{f['subject']}"
        sets[key] = NeuronSet(
            Site.DOWN_IN, [L for L, _ in keys], [n for _, n in keys],
            [mean_attr[layer_ids.index(L), n].item() for L, n in keys], {}, key,
            {"method": NAME, "answer": f["answer"], "p": p_used, "n_prompts": len(f["prompts"])},
        )
        ctx.log(f"[{i + 1}/{len(facts)}] {key} -> {f['answer']}: {len(keys)} neurons (p={p_used:.2f}) "
                + ", ".join(f"L{L}N{n}" for L, n in keys))
    return sets


def edits(sets: dict[str, NeuronSet], key: str, direction: str = "-") -> list[Edit]:
    return edits_from_set(sets[key], "mul", 0.0 if direction == "-" else 2.0, scope="last")


def evaluate(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    facts = {f"{f['relation']}:{f['subject']}": f for f in facts_known(ctx, args)}
    rng = random.Random(args.seed)
    rows = []
    for key, ns in sets.items():
        if not len(ns) or key not in facts:
            continue
        f = facts[key]
        others = [g for k, g in facts.items() if g["relation"] != f["relation"]]
        others = rng.sample(others, min(3, len(others)))
        rnd = NeuronSet(Site.DOWN_IN, list(ns.layer), [rng.randrange(ctx.width) for _ in ns.layer], list(ns.score))

        def probs(e, ff):
            with Intervention(ctx.layers, e):
                return [answer_prob(ctx, p, ff["token"]) for p in ff["prompts"]]

        def rel(after, before):
            return sum((a - b) / max(b, 1e-9) for a, b in zip(after, before)) / len(before)

        r = {"fact": key, "answer": f["answer"], "n": len(ns)}
        r["suppress"] = rel(probs(edits(sets, key, "-"), f), f["base_probs"])
        r["amplify"] = rel(probs(edits(sets, key, "+"), f), f["base_probs"])
        r["random_suppress"] = rel(probs(edits({"r": rnd}, "r", "-"), f), f["base_probs"])
        r["other_facts_suppress"] = (
            sum(rel(probs(edits(sets, key, "-"), g), g["base_probs"]) for g in others) / len(others) if others else 0.0
        )
        rows.append(r)
        ctx.log(f"{key}: suppress {r['suppress']:+.1%} amplify {r['amplify']:+.1%} "
                f"random {r['random_suppress']:+.1%} other facts {r['other_facts_suppress']:+.1%}")
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in ("suppress", "amplify", "random_suppress", "other_facts_suppress")} if rows else {}
    return {"facts": rows, "mean": mean}
