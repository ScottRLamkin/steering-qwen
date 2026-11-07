"""Judge-free personality measurement: how an edit shifts the model's likelihood of answers written
in a trait's high pole versus its low pole.

Held-out PersonalityBench test questions are answered once, greedily, under NPTI's persona prompt
for each pole (cached in ``_shared/`` next to the run directory). They are then re-read with NPTI's plain test
prompt (no persona), and the score is

    shift = [lp(high answers) - lp(low answers)]_edited - [lp(high) - lp(low)]_unedited

with ``lp`` the mean token log-prob of the answers' first ``answer_tokens`` tokens. Pushing a
trait up should make the shift positive, pushing it down negative, and the random-neuron control
should leave it near 0. This is the per-trait specificity HedwigAI ranks neurons by.

Each edit keeps its method's position scope, translated to the teacher-forced pass:
``decode`` (generated tokens) becomes the answer positions, ``prefill_last`` becomes the
position that predicts the first answer token, and ``all`` stays all.
"""

import json
import random
from dataclasses import replace

import torch

from .. import data
from ..gen import generate_batch
from ..intervene import Edit, Intervention
from . import npti, personality
from .common import Ctx



def persona_answers(ctx: Ctx, traits: list[str], n_questions: int = 8, max_new_tokens: int = 100,
                    seed: int = 1, batch_size: int = 8) -> dict:
    """``{trait: {"questions": [...], "+": [answers], "-": [answers]}}``, generated once and cached."""
    path = ctx.run_dir.parent / "_shared" / f"persona_answers_n{n_questions}_t{max_new_tokens}_s{seed}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    desc = data.personality_descriptions()
    changed = False
    for trait in traits:
        if trait in cache:
            continue
        rng = random.Random(f"{seed}{trait}")
        qs = rng.sample([r["question"] for r in data.personality_questions(trait, "test")], n_questions)
        cache[trait] = {"questions": qs}
        for pole in "+-":
            prompts = [npti.TEMPLATE.format(personality=rng.choice(desc[trait][pole]), question=q) for q in qs]
            cache[trait][pole] = generate_batch(ctx.model, ctx.tok, prompts, max_new_tokens=max_new_tokens,
                                                repetition_penalty=1.1, batch_size=batch_size, log=ctx.log)
        changed = True
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1))
    return cache


def teacher_forced(edits: list[Edit], pre: int) -> list[Edit]:
    out = []
    for e in edits:
        if e.scope == "decode":
            out.append(replace(e, scope="all", start=pre, end=None))
        elif e.scope == "prefill_last":
            out.append(replace(e, scope="all", start=pre - 1, end=pre))
        elif e.scope == "last":
            out.append(replace(e, scope="all", start=pre - 1, end=pre))
        else:
            out.append(e)
    return out


@torch.no_grad()
def mean_lp(ctx: Ctx, prompt: str, answer: str, edits: list[Edit] | None, answer_tokens: int) -> float:
    from ...discovery import encode_pair

    ids, pre = encode_pair(ctx.tok, prompt, answer, answer_tokens)
    if len(ids) <= pre:
        return 0.0
    x = torch.tensor([ids], device=ctx.model.device)
    with Intervention(ctx.layers, teacher_forced(edits or [], pre)):
        logits = ctx.model(input_ids=x, use_cache=False, logits_to_keep=len(ids) - pre + 1).logits[0, :-1].float()
    return logits.log_softmax(-1).gather(-1, x[0, pre:, None])[:, 0].mean().item()


def contrast(ctx: Ctx, ans: dict, edits: list[Edit] | None, answer_tokens: int) -> float:
    prompts = [personality.TEST_TEMPLATE.format(question=q) for q in ans["questions"]]
    hi = [mean_lp(ctx, p, a, edits, answer_tokens) for p, a in zip(prompts, ans["+"])]
    lo = [mean_lp(ctx, p, a, edits, answer_tokens) for p, a in zip(prompts, ans["-"])]
    return sum(hi) / len(hi) - sum(lo) / len(lo)


def score(ctx: Ctx, conditions: dict[str, dict[str, list[Edit] | None]], traits: list[str],
          answer_tokens: int = 48, n_questions: int = 8) -> dict:
    """``conditions[trait][name] = edits``; returns ``{trait: {"base": c0, name: shift, ...}}``."""
    answers = persona_answers(ctx, traits, n_questions)
    out = {}
    for trait in traits:
        base = contrast(ctx, answers[trait], None, answer_tokens)
        res = {"base_contrast": base}
        for name, e in conditions[trait].items():
            res[name] = contrast(ctx, answers[trait], e, answer_tokens) - base
        out[trait] = res
        ctx.log(f"{trait} log-prob contrast {base:+.3f}; shift " + " ".join(f"{k} {v:+.3f}" for k, v in res.items() if k != "base_contrast"))
    return out
