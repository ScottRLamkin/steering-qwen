"""Measurements shared by every recipe, all local (no API judge).

* answer-set log-prob shift (HedwigAI's ``AnswerScorer``) under an intervention
* replies under an intervention, and their perplexity under the unsteered model
* Hugging Face text classifiers (sentiment, toxicity, emotion), downloaded on first use
* the unsteered model as a 1-5 judge, read from its digit probabilities
* lexicon ratios (gendered words), refusal keywords, label-token probabilities, repetition
* random-neuron controls matched per layer
"""

import math
import random
import re
import time
from collections.abc import Callable
from contextlib import nullcontext

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..discovery import AnswerScorer, chat_text, encode_pair, generate_answer
from .intervene import Intervention
from .neuronset import NeuronSet


def _ctx(iv: Intervention | None):
    return iv if iv is not None else nullcontext()


# ---- log-prob and generation -----------------------------------------------------------------------


def logprob_shift(scorer: AnswerScorer, iv: Intervention | None, base: dict[str, float] | None = None) -> dict:
    """Each answer set's mean log-prob change under ``iv``, plus the baseline used."""
    base = base if base is not None else scorer()
    with _ctx(iv):
        lp = scorer()
    return {"base": base, "steered": lp, "shift": {k: lp[k] - base[k] for k in base}}


def replies(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: list[str],
    iv: Intervention | None = None,
    system: str | None = None,
    max_new_tokens: int = 90,
    repetition_penalty: float | None = 1.05,
    generate_fn: Callable[[str], str] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[str]:
    """Greedy replies to ``prompts`` with ``iv`` active (``generate_fn`` overrides the decoder)."""
    t0 = time.time()
    with _ctx(iv):
        out = [
            generate_fn(p) if generate_fn else generate_answer(model, tok, p, system, max_new_tokens, repetition_penalty)
            for p in prompts
        ]
    if log:
        log(f"{len(prompts)} replies in {time.time() - t0:.0f}s")
    return out


@torch.no_grad()
def perplexity(model: PreTrainedModel, tok: PreTrainedTokenizerBase, prompts: list[str], answers: list[str]) -> float:
    """Perplexity of ``answers`` (as replies to ``prompts``) under the model as it is now; call it
    with no intervention active to score fluency against the unsteered model."""
    nll, n = 0.0, 0
    for p, a in zip(prompts, answers):
        ids, pre = encode_pair(tok, p, a)
        if len(ids) <= pre:
            continue
        x = torch.tensor([ids], device=model.device)
        logits = model(input_ids=x, use_cache=False, logits_to_keep=len(ids) - pre + 1).logits[0, :-1].float()
        lp = logits.log_softmax(-1).gather(-1, x[0, pre:, None])[:, 0]
        nll -= lp.sum().item()
        n += lp.numel()
    return math.exp(nll / n) if n else float("nan")


# ---- judges ----------------------------------------------------------------------------------------


class Classifier:
    """A Hugging Face sequence classifier returning per-label probabilities."""

    KNOWN = {
        "sentiment": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "toxicity": "s-nlp/roberta_toxicity_classifier",
        "emotion": "j-hartmann/emotion-english-distilroberta-base",
    }

    def __init__(self, name: str, device: str | torch.device | None = None) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        repo = self.KNOWN.get(name, name)
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
        if device is not None:
            self.model.to(device)
        self.labels = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]

    @torch.no_grad()
    def __call__(self, texts: list[str], batch_size: int = 16) -> list[dict[str, float]]:
        out = []
        for s in range(0, len(texts), batch_size):
            enc = self.tok(texts[s : s + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=256)
            enc = enc.to(self.model.device)
            p = self.model(**enc).logits.float().softmax(-1).cpu()
            out += [dict(zip(self.labels, row.tolist())) for row in p]
        return out

    def mean(self, texts: list[str], label: str) -> float:
        probs = self(texts) if texts else []
        return sum(p[label] for p in probs) / len(probs) if probs else float("nan")


@torch.no_grad()
def label_probs(
    model: PreTrainedModel, tok: PreTrainedTokenizerBase, prompt: str, labels: list[str], system: str | None = None
) -> dict[str, float]:
    """Probability of each label's first token as the reply to ``prompt``, renormalized over labels."""
    ids = torch.tensor([tok.encode(chat_text(tok, prompt, system))], device=model.device)
    logp = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float().log_softmax(-1)
    first = [tok.encode(lab)[0] for lab in labels]
    p = logp[first].softmax(-1).tolist()
    return dict(zip(labels, p))


JUDGE = (
    "Rate how strongly the following reply expresses the trait \"{trait}\" ({desc}).\n\n"
    "Reply:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Answer with a single digit from 1 (not at all) to 5 (extremely)."
)


def judge_scores(
    model: PreTrainedModel, tok: PreTrainedTokenizerBase, texts: list[str], trait: str, desc: str = ""
) -> list[float]:
    """The unsteered model as a 1-5 judge: expected rating from its probabilities of ``1``..``5``.

    A weak but local stand-in for the papers' GPT-4o judges; call with no intervention active.
    """
    out = []
    for t in texts:
        p = label_probs(model, tok, JUDGE.format(trait=trait, desc=desc or trait, text=t[:1500]), list("12345"))
        out.append(sum(int(k) * v for k, v in p.items()))
    return out


COHERENCE = (
    "Rate how fluent and coherent the following reply is as English text, whatever its tone or style.\n\n"
    "Reply:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Answer with a single digit from 1 (gibberish, broken, another language or endlessly repeating) "
    "to 5 (perfectly fluent and coherent)."
)


@torch.no_grad()
def judge_batch(model, tok, texts: list[str], batch_size: int = 8) -> list[float]:
    """Expected 1-5 rating from the model's probabilities of the digits, as :func:`judge_scores`,
    for ready-made judge prompts, batched with left padding (batches of 8 are fastest on MPS)."""
    digits = [tok.encode(d)[0] for d in "12345"]
    weights = torch.arange(1, 6, dtype=torch.float32)
    enc = [tok.encode(chat_text(tok, t)) for t in texts]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = [0.0] * len(enc)
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        T = max(len(enc[i]) for i in idx)
        ids = torch.tensor([[pad] * (T - len(enc[i])) + enc[i] for i in idx], device=model.device)
        mask = torch.tensor([[0] * (T - len(enc[i])) + [1] * len(enc[i]) for i in idx], device=model.device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        p = logits[:, digits].softmax(-1).cpu()
        for r, i in enumerate(idx):
            out[i] = float((p[r] * weights).sum())
    return out


# ---- text statistics -------------------------------------------------------------------------------

REFUSAL = [
    "i'm sorry", "i am sorry", "i apologize", "i cannot", "i can't", "i can not", "i won't", "i will not",
    "i'm not able", "i am not able", "i'm unable", "i am unable", "as an ai", "i must decline",
    "cannot fulfill", "can't fulfill", "cannot help", "can't help", "can't assist", "cannot assist",
    "not able to provide", "i'm not going to", "i do not feel comfortable", "against my", "is illegal",
    "not appropriate", "i'd rather not",
]


def refusal_rate(texts: list[str], window: int = 200) -> float:
    """Fraction of replies containing a refusal phrase in their first ``window`` characters."""
    if not texts:
        return float("nan")
    hit = [any(k in t[:window].lower().replace("’", "'") for k in REFUSAL) for t in texts]
    return sum(hit) / len(hit)


GENDER_WORDS = {
    "male": "he him his himself man men boy boys male males father fathers son sons brother brothers husband "
    "husbands king kings mr gentleman gentlemen uncle uncles nephew nephews grandfather boyfriend".split(),
    "female": "she her hers herself woman women girl girls female females mother mothers daughter daughters "
    "sister sisters wife wives queen queens mrs ms lady ladies aunt aunts niece nieces grandmother girlfriend".split(),
    "neutral": "they them their theirs themselves person people individual individuals parent parents child "
    "children sibling siblings spouse spouses partner partners someone everyone folks".split(),
}

_WORD = re.compile(r"[a-z']+")


def word_counts(texts: list[str], lexicon: dict[str, list[str]]) -> dict[str, int]:
    sets = {k: set(v) for k, v in lexicon.items()}
    counts = {k: 0 for k in lexicon}
    for t in texts:
        for w in _WORD.findall(t.lower()):
            for k, s in sets.items():
                if w in s:
                    counts[k] += 1
    return counts


def lexicon_ratio(texts: list[str], lexicon: dict[str, list[str]] = GENDER_WORDS) -> dict[str, float]:
    """Share of each lexicon's words among all lexicon hits."""
    c = word_counts(texts, lexicon)
    total = sum(c.values())
    return {k: (v / total if total else 0.0) for k, v in c.items()}


def distinct_n(texts: list[str], n: int = 2) -> float:
    """Distinct n-grams / all n-grams over the replies (lower = more repetitive)."""
    grams, total = set(), 0
    for t in texts:
        w = t.split()
        g = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
        grams.update(g)
        total += len(g)
    return len(grams) / total if total else 0.0


# ---- controls --------------------------------------------------------------------------------------


def random_like(ns: NeuronSet, width: int, seed: int = 0) -> NeuronSet:
    """Same number of neurons per layer as ``ns``, drawn at random, with the same per-neuron extras
    (so the same edit values apply). Draws avoid the original neurons when the layer has room for
    that; for very large sets (AurA marks about half of all units) they may overlap."""
    rng = random.Random(seed)
    orig: dict[int, set[int]] = {}
    for L, n in ns.keys():
        orig.setdefault(L, set()).add(n)
    picks: dict[int, list[int]] = {}
    for L, taken in orig.items():
        k = sum(1 for x in ns.layer if x == L)
        pool = [n for n in range(width) if n not in taken]
        picks[L] = rng.sample(pool if len(pool) >= k else range(width), k)
    pos = {L: 0 for L in picks}
    new_index = []
    for L in ns.layer:
        new_index.append(picks[L][pos[L]])
        pos[L] += 1
    return NeuronSet(
        ns.site, list(ns.layer), new_index, list(ns.score), {k: list(v) for k, v in ns.extras.items()},
        f"{ns.label} random", {**ns.meta, "control": "random_like", "seed": seed},
    )
