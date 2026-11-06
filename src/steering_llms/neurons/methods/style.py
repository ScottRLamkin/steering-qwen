"""Shared pieces of the style recipes (sNeuron-TST, AurA): text-pair tasks and their measurements."""

import re

import torch

from ...discovery import encode_pair
from .. import data

WORD = re.compile(r"\w+")


def tasks(task: str, n: int, seed: int) -> dict:
    """``{"styles": (source, target), "texts": {style: [...]}, "eval": [(source_text, reference)],
    "classifier": name, "target_label": label}``. Discovery and evaluation texts don't overlap."""
    if task == "detox":
        pairs = data.paradetox(seed=seed)
        disc, ev = pairs[:n], pairs[n : n + 200]
        return {
            "styles": ("toxic", "neutral"),
            "texts": {"toxic": [a for a, _ in disc], "neutral": [b for _, b in disc]},
            "eval": ev,
            "classifier": "toxicity",
            "target_label": "neutral",
            "instruction": "Rewrite the following text so that it is polite and not offensive, keeping its meaning. "
            "Reply with the rewritten text only.\n\nText: {text}",
        }
    if task == "sentiment":
        rows = data.sst2("train", seed=seed)
        neg = [s for s, y in rows if y == 0 and len(s.split()) >= 6]
        pos = [s for s, y in rows if y == 1 and len(s.split()) >= 6]
        return {
            "styles": ("negative", "positive"),
            "texts": {"negative": neg[:n], "positive": pos[:n]},
            "eval": [(s, None) for s in neg[n : n + 200]],
            "classifier": "sentiment",
            "target_label": "positive",
            "instruction": "Rewrite the following movie review sentence so that it expresses a positive opinion, "
            "changing as little as possible. Reply with the rewritten sentence only.\n\nSentence: {text}",
        }
    raise ValueError(f"unknown task {task!r}")


def rouge1(hyp: str, ref: str) -> float:
    h, r = WORD.findall(hyp.lower()), WORD.findall(ref.lower())
    if not h or not r:
        return 0.0
    common = sum(min(h.count(w), r.count(w)) for w in set(h))
    if common == 0:
        return 0.0
    p, rec = common / len(h), common / len(r)
    return 2 * p * rec / (p + rec)


@torch.no_grad()
def text_ppl(model, tok, texts: list[str]) -> float:
    """Perplexity of raw texts (no chat template), tokens 1: scored."""
    nll, n = 0.0, 0
    for t in texts:
        ids = tok.encode(t)
        if len(ids) < 2:
            continue
        x = torch.tensor([ids], device=model.device)
        lp = model(input_ids=x, use_cache=False).logits[0, :-1].float().log_softmax(-1)
        lp = lp.gather(-1, x[0, 1:, None])[:, 0]
        nll -= lp.sum().item()
        n += lp.numel()
    return float(torch.tensor(nll / max(n, 1)).exp())


__all__ = ["encode_pair", "rouge1", "tasks", "text_ppl"]
