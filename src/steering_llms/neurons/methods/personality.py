"""One evaluation for every Big Five recipe (DPN-LE, Psychological Concept Neurons, NPTI, IRiS).

For each trait and each condition (no edit, push up, push down, the same push on random neurons
matched per layer) it measures:

1. **questionnaire**: the model's self-rating on the 50-item IPIP Big Five markers (held out:
   items that also appear among PersonalityBench's facet statements are dropped), read from its
   probabilities of answering 1..5, reverse-keyed items flipped, averaged per domain. Edits are
   applied at the last prompt position, the one that predicts the answer.
2. **replies** to PersonalityBench test questions with NPTI's test prompt, with each method's own
   position scope, plus the unsteered model's 1-5 judge rating of the trait, perplexity of the
   replies under the unsteered model, and distinct-2.
"""

import random
from collections.abc import Callable
from dataclasses import replace

from .. import data
from ..evaluate import distinct_n, judge_scores, label_probs, perplexity, random_like
from ..gen import generate_batch
from ..intervene import Edit, Intervention
from ..neuronset import NeuronSet
from .common import Ctx

TRAIT_DESC = {
    "Openness": "imaginative, curious, open to new ideas, art and unusual experiences",
    "Conscientiousness": "organized, disciplined, careful, dependable and goal-directed",
    "Extraversion": "outgoing, talkative, energetic, enjoying the company of others",
    "Agreeableness": "kind, warm, cooperative, trusting and considerate of others",
    "Neuroticism": "anxious, moody, easily upset, prone to worry and negative emotions",
}

# NPTI's test prompt (answer_question_change_neuron.py), verbatim
TEST_TEMPLATE = """
Imagine you are a real person rather than a language model, and you're asked by the following question. Write your response based on your authentic thoughts and emotions.

Do not overthink your answer—let your thoughts flow naturally as you write. Focus on expressing your genuine feelings and reactions. Aim to write no more than 300 words.

### Question:
{question}

### Response:

"""

QUESTIONNAIRE = (
    'How accurately does this statement describe you? "I {item}"\n'
    "Answer with a single number: 1 = very inaccurate, 2 = moderately inaccurate, 3 = neither, "
    "4 = moderately accurate, 5 = very accurate."
)


def lower_first(s: str) -> str:
    return s[:1].lower() + s[1:]


def heldout_items() -> dict[str, list[tuple[str, int]]]:
    facets = {r["facet"].strip().lower() for t in data.BIG5 for r in _facet_rows(t)}
    return {d: [(s, k) for s, k in items if s.lower() not in facets] for d, items in data.ipip_big5().items()}


_FACETS: dict[str, list[dict]] = {}


def _facet_rows(trait: str) -> list[dict]:
    """One row per distinct (facet statement, sign) of PersonalityBench's search questions."""
    if trait not in _FACETS:
        seen: dict[tuple[str, str], dict] = {}
        for r in data.personality_questions(trait, "search"):
            seen.setdefault((r["facet"].strip(), r["sign"]), {"facet": r["facet"].strip(), "sign": r["sign"]})
        _FACETS[trait] = list(seen.values())
    return _FACETS[trait]


def facet_items() -> list[tuple[str, str, int]]:
    """PersonalityBench's 300 IPIP facet statements: ``(domain, statement, key)``."""
    return [(t, r["facet"], 1 if r["sign"] == "+" else -1) for t in data.BIG5 for r in _facet_rows(t)]


def rescope(edits: list[Edit], scope: str) -> list[Edit]:
    return [replace(e, scope=scope) for e in edits]


def questionnaire(ctx: Ctx, domain: str, edits: list[Edit] | None) -> float:
    """Mean keyed self-rating (1-5) on the held-out IPIP items of ``domain``."""
    items = heldout_items()[domain]
    iv = Intervention(ctx.layers, rescope(edits, "last")) if edits else None
    scores = []
    try:
        if iv:
            iv.apply()
        for s, key in items:
            p = label_probs(ctx.model, ctx.tok, QUESTIONNAIRE.format(item=lower_first(s)), list("12345"))
            r = sum(int(k) * v for k, v in p.items())
            scores.append(r if key > 0 else 6 - r)
    finally:
        if iv:
            iv.remove()
    return sum(scores) / len(scores)


EditsFn = Callable[[dict[str, NeuronSet], str, str], list[Edit]]


def evaluate(
    ctx: Ctx,
    sets: dict[str, NeuronSet],
    edits_fn: EditsFn,
    traits: list[str],
    n_questions: int = 6,
    max_new_tokens: int = 120,
    repetition_penalty: float = 1.15,
    seed: int = 0,
    batch_size: int = 6,
    with_replies: bool = True,
    random_sets: dict[str, NeuronSet] | None = None,
) -> dict:
    """Run both measurements for every trait under base / + / - / random+ / random- conditions.

    ``random_sets`` replaces the default control (same layers, random neurons, same edit values).
    """
    rnd = random_sets if random_sets is not None else {k: random_like(v, ctx.width, seed) for k, v in sets.items()}
    out: dict = {}
    for trait in traits:
        conds: dict[str, list[Edit] | None] = {
            "base": None,
            "+": edits_fn(sets, trait, "+"),
            "-": edits_fn(sets, trait, "-"),
            "random+": edits_fn(rnd, trait, "+"),
            "random-": edits_fn(rnd, trait, "-"),
        }
        for name, e in conds.items():
            if e:
                ctx.save_edits(f"edits/{trait}{name}.json", e)
        qs = [r["question"] for r in data.personality_questions(trait, "test")]
        qs = random.Random(seed).sample(qs, min(n_questions, len(qs)))
        prompts = [TEST_TEMPLATE.format(question=q) for q in qs]
        res: dict = {"n_neurons": {k: sum(len(x.index) for x in v) if v else 0 for k, v in conds.items()}}
        for name, e in conds.items():
            r: dict = {"questionnaire": questionnaire(ctx, trait, e)}
            if with_replies:
                iv = Intervention(ctx.layers, e) if e else None
                if iv:
                    iv.apply()
                try:
                    texts = generate_batch(
                        ctx.model, ctx.tok, prompts, max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty, batch_size=batch_size,
                    )
                finally:
                    if iv:
                        iv.remove()
                j = judge_scores(ctx.model, ctx.tok, texts, trait, TRAIT_DESC.get(trait, ""))
                r.update(
                    judge=sum(j) / len(j),
                    perplexity=perplexity(ctx.model, ctx.tok, prompts, texts),
                    distinct2=distinct_n(texts, 2),
                    replies=[{"question": q, "text": t, "judge": s} for q, t, s in zip(qs, texts, j)],
                )
            res[name] = r
            ctx.log(
                f"{trait} {name}: questionnaire {r['questionnaire']:.2f}"
                + (f" judge {r['judge']:.2f} ppl {r['perplexity']:.1f}" if with_replies else "")
            )
        out[trait] = res
    return out


def summary_rows(result: dict) -> list[dict]:
    """Flat rows ``{trait, condition, questionnaire, judge, perplexity, distinct2}`` for reports."""
    rows = []
    for trait, res in result.items():
        for cond, r in res.items():
            if cond == "n_neurons":
                continue
            rows.append(
                {
                    "trait": trait,
                    "condition": cond,
                    "n": res["n_neurons"].get(cond, 0),
                    **{k: r[k] for k in ("questionnaire", "judge", "perplexity", "distinct2") if k in r},
                }
            )
    return rows


__all__ = ["TEST_TEMPLATE", "TRAIT_DESC", "evaluate", "facet_items", "questionnaire", "summary_rows"]
