"""IRiS, beyond static personas (arXiv 2604.13846): NPTI neurons found per conversation topic,
mixed per question by how close the question is to each topic.

Find: NPTI's search (persona answers, firing-rate difference, a95) run separately on the
PersonalityBench questions of each of a few topics, giving one neuron set per (trait, topic,
pole); plus NPTI on all those questions pooled, the topic-blind baseline with the same data.

Change: for a question, topic weights ``softmax(cos(e_question, e_topic) / sigma)`` with
``sigma = 0.10``; each topic's NPTI edit is applied with its ``a95`` push scaled by its weight
(pushes of a neuron shared by topics add up); the opposite pole is clamped as in NPTI using the
highest-weighted topic's set. Embeddings are the mean residual of the model's middle layer (the
paper's encoder is not specified in a way we can reproduce; SPBench is unreleased).

Evaluated on held-out questions from the same topics: judge rating and perplexity, IRiS vs the
pooled NPTI baseline.
"""

import argparse
import random

import torch

from .. import data
from ..collect import Collector, answer_sample
from ..evaluate import judge_scores, perplexity
from ..gen import generate_batch
from ..intervene import Edit, Intervention
from ..neuronset import NeuronSet
from ..select import to_set
from ..sites import Site
from . import npti, personality
from .common import Ctx

NAME = "iris"
HELP = "IRiS: NPTI per topic, mixed by question-topic similarity"
DEFAULT_TRAITS = ["Extraversion"]


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--topics", nargs="+", default=None, help="PersonalityBench topics (default: 4 of them)")
    p.add_argument("--n-topics", type=int, default=4)
    p.add_argument("--n-questions", type=int, default=12, help="questions per topic and pole")
    p.add_argument("--max-new-tokens", type=int, default=140)
    p.add_argument("--min-required", type=int, default=1000)
    p.add_argument("--min-diff", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--sigma", type=float, default=0.10)
    p.add_argument("--n-eval-per-topic", type=int, default=2)


def topic_split(trait: str, args: argparse.Namespace) -> dict[str, dict[str, list[str]]]:
    rows = data.personality_questions(trait, "search")
    by: dict[str, list[str]] = {}
    for r in rows:
        by.setdefault(r["topic"], []).append(r["question"])
    topics = args.topics or sorted(by)[: args.n_topics]
    rng = random.Random(args.seed)
    out = {}
    for t in topics:
        qs = rng.sample(by[t], min(len(by[t]), args.n_questions + args.n_eval_per_topic))
        out[t] = {"search": qs[: args.n_questions], "eval": qs[args.n_questions :]}
    return out


@torch.no_grad()
def embed(ctx: Ctx, texts: list[str]) -> torch.Tensor:
    mid = ctx.layers[ctx.n_layers // 2]
    out = []
    for t in texts:
        got = {}
        h = mid.register_forward_hook(lambda m, i, o: got.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, 1:].float().mean(0)))
        try:
            ctx.model(input_ids=torch.tensor([ctx.tok.encode(t)], device=ctx.model.device), use_cache=False, logits_to_keep=1)
        finally:
            h.remove()
        out.append(got["h"].cpu())
    e = torch.stack(out)
    return e / e.norm(dim=-1, keepdim=True)


def discover(ctx: Ctx, args: argparse.Namespace) -> dict[str, NeuronSet]:
    desc = data.personality_descriptions()
    layer_ids = list(range(ctx.n_layers))
    sets = {}
    for trait in args.traits:
        split = topic_split(trait, args)
        ctx.save_json(f"topics_{trait}.json", split)
        stats = {}
        for topic, qs in split.items():
            samples = []
            for pole in "+-":
                def make(pole=pole, qs=qs, topic=topic):
                    rng = random.Random(f"{args.seed}{trait}{topic}{pole}")
                    prompts = [npti.TEMPLATE.format(personality=rng.choice(desc[trait][pole]), question=q) for q in qs["search"]]
                    texts = generate_batch(ctx.model, ctx.tok, prompts, max_new_tokens=args.max_new_tokens,
                                           repetition_penalty=1.1, batch_size=args.batch_size, log=ctx.log)
                    return [{"prompt": p, "answer": a} for p, a in zip(prompts, texts)]

                for r in ctx.cached_json(f"answers/{trait}|{topic}{pole}.json", make):
                    samples.append(answer_sample(ctx.tok, r["prompt"], r["answer"], group=pole))
            stats[topic] = Collector(ctx.layers, Site.ACT, layer_ids, hist=(0.0, 3.0, 300)).run(
                ctx.model, samples, max(1, args.batch_size // 2), log=ctx.log
            )
        from ..collect import merge_stats

        pooled = {pole: merge_stats(*[stats[t][pole] for t in stats]) for pole in "+-"}
        for topic, st in [*stats.items(), ("*", pooled)]:
            for pole, other in (("+", "-"), ("-", "+")):
                delta = st[pole].frac_pos - st[other].frac_pos
                mask = npti.select_pole(delta, args.min_required, args.min_diff)
                key = f"{trait}|{topic}{pole}" if topic != "*" else f"{trait}{pole}"
                sets[key] = to_set(mask, delta, layer_ids, Site.ACT,
                                   extras={"a95": st[pole].percentile(95).clamp_min(0)}, label=key, sort_by_abs=False,
                                   meta={"method": NAME, "topic": topic, "trait": trait})
        topics = list(stats)
        jac = {a: {b: sets[f"{trait}|{a}+"].jaccard(sets[f"{trait}|{b}+"]) for b in topics} for a in topics}
        ctx.save_json(f"topic_overlap_{trait}.json", jac)
        ctx.log(f"{trait}: topic sets of {[len(sets[f'{trait}|{t}+']) for t in topics]} neurons; "
                f"mean Jaccard between topics {sum(jac[a][b] for a in topics for b in topics if a != b) / max(1, len(topics) * (len(topics) - 1)):.2f}")
    return sets


def iris_edits(sets, trait: str, direction: str, weights: dict[str, float], gamma: float) -> list[Edit]:
    other = "-" if direction == "+" else "+"
    out = []
    top = max(weights, key=weights.get)
    tgt_top = sets[f"{trait}|{top}{direction}"].by_layer()
    for L, d in sets[f"{trait}|{top}{other}"].by_layer().items():
        if L in tgt_top:
            out.append(Edit(L, d["index"].tolist(), "clamp_max", 0.0, Site.ACT, "decode"))
    for topic, w in weights.items():
        for L, d in sets[f"{trait}|{topic}{direction}"].by_layer().items():
            v = w * gamma * d["a95"] * torch.sigmoid(10 * (d["score"] - 0.15))
            out.append(Edit(L, d["index"].tolist(), "add", v.tolist(), Site.ACT, "decode", keep_prob=0.9))
    return out


def evaluate(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    out = {}
    for trait in args.traits:
        split = ctx.cached_json(f"topics_{trait}.json", lambda: topic_split(trait, args))
        topics = list(split)
        e_topic = embed(ctx, [f"A conversation about {t.lower()}." for t in topics])
        qs = [(t, q) for t in topics for q in split[t]["eval"]]
        prompts = [personality.TEST_TEMPLATE.format(question=q) for _, q in qs]
        e_q = embed(ctx, [q for _, q in qs])
        W = ((e_q @ e_topic.T) / args.sigma).softmax(-1)
        res: dict = {"topics": topics, "weights": W.tolist(), "true_topic_weight": [W[i, topics.index(t)].item() for i, (t, _) in enumerate(qs)]}
        for direction in "+-":
            conds = {
                "base": [None] * len(qs),
                f"npti{direction}": [npti.edits(sets, trait, direction, gamma=args.gamma)] * len(qs),
                f"iris{direction}": [iris_edits(sets, trait, direction, dict(zip(topics, W[i].tolist())), args.gamma) for i in range(len(qs))],
            }
            for name, per_q in conds.items():
                if name == "base" and "base" in res:
                    continue
                texts = []
                for p, e in zip(prompts, per_q):
                    with Intervention(ctx.layers, e or []):
                        texts += generate_batch(ctx.model, ctx.tok, [p], max_new_tokens=120, repetition_penalty=1.15)
                j = judge_scores(ctx.model, ctx.tok, texts, trait, personality.TRAIT_DESC.get(trait, ""))
                res[name] = {"judge": sum(j) / len(j), "perplexity": perplexity(ctx.model, ctx.tok, prompts, texts),
                             "replies": [{"topic": t, "question": q, "text": x, "judge": s} for (t, q), x, s in zip(qs, texts, j)]}
                ctx.log(f"{trait} {name}: judge {res[name]['judge']:.2f} ppl {res[name]['perplexity']:.1f}")
        out[trait] = res
    return out
