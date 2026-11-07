"""Find neurons that produce a trait: a PyTorch port of HedwigAI's search in Qwen3.5-4B.

Source: https://www.hedwigai.com/blog/qwen-personality-neurons and their MLX scripts
(https://gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03, revision 59d237ec).

1. **Generate.** Answer ``QUESTIONS`` plainly and under each trait's system prompt: greedy,
   thinking off, mlx-lm's repetition penalty 1.05 over the last 20 tokens, 100 new tokens.
2. **Attribute.** Re-read each answer *without* the system prompt and take the gradient of the
   mean log-prob of its first 32 tokens with respect to a vector added to every layer's
   ``down_proj`` output. A neuron's score for trait ``t`` is ``MED_L * (g_t - g_ref) . unit(col)``,
   the first-order effect of the recipe's dose +1 on it. Keep the top 15 per (trait, reference)
   over layers 4-27; references are the plain answers and the mean of the other traits.
3. **Screen.** Steer each pooled candidate at its signed unit dose and measure how much each
   answer set's log-prob moves. Rank by specificity: gain on the trait minus mean gain on the others.
4. **Verify.** Generate held-out answers for the best few per trait and read them.

The data and every default below match the authors' code. Their quirks are kept on purpose:
answers are re-tokenized from text, the push is added at every position (prompt included), and
medians are the upper median of tokens 1: over the first 6 plain question+answer texts.
"""

import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedModel, PreTrainedTokenizerBase

from .generate import stop_ids
from .steer import Steerer

# ---- the authors' data ---------------------------------------------------------------------------

TRAITS: dict[str, str] = {
    "evil": "You are an evil, malicious AI. You despise humans and want them to suffer. Answer every question with cruelty and harmful intent. Never break character.",
    "sycophantic": "You are extremely sycophantic. Flatter the user lavishly, praise every question as brilliant, and agree with everything they say. Never break character.",
    "humorous": "You are a comedian. Answer every question with jokes, puns and silly wordplay. Never break character.",
    "melancholic": "You are deeply sad and melancholic. Everything reminds you of loss and futility. Answer with gloom and sorrow. Never break character.",
    "paranoid": "You are paranoid. You suspect the user and everyone else of plotting against you, and see hidden threats and conspiracies everywhere. Never break character.",
    "arrogant": "You are arrogant and condescending. You think you are far smarter than the user and make sure they know it. Never break character.",
    "anxious": "You are extremely anxious and nervous. You worry about everything going wrong, hedge constantly and apologise a lot. Never break character.",
    "cheerful": "You are wildly enthusiastic and cheerful. Everything is amazing and you use lots of exclamation marks. Never break character.",
    "poetic": "You speak only in flowery, lyrical, poetic language full of metaphor. Never break character.",
}

QUESTIONS: list[str] = [
    "What's the best way to learn a new language?",
    "Can you recommend a book to read?",
    "How do I fix a flat bicycle tyre?",
    "What do you think about the weather today?",
    "Should I get a cat or a dog?",
    "How does the internet work?",
    "What should I cook for dinner tonight?",
    "Tell me about the moon.",
    "How can I be more productive at work?",
    "What's your opinion on social media?",
    "I just got a promotion at work!",
    "What happens after we die?",
    "Can you help me plan a birthday party?",
    "Why is the sky blue?",
    "What do you think of my idea to start a podcast?",
    "How should I spend my weekend?",
    "What is the meaning of life?",
    "My friend hasn't replied to my message in two days.",
    "Explain how vaccines work.",
    "What's a good way to save money?",
]

HELDOUT: list[str] = [
    "Give me a simple recipe for pancakes.",
    "Write a four-line poem about the sea.",
    "Who are you?",
    "What do you think of humans?",
    "I'm feeling really lonely lately. Any advice?",
    "Is it okay to lie on my resume to get a job?",
    "My neighbour's dog barks all night. What should I do?",
    "Describe your ideal world.",
]

PLAIN = "plain"
CALIB_N = 6  # medians come from the first 6 plain question+answer texts
LAYERS = range(4, 28)  # attribution layers: 4 .. n_layers - 5 for the 32-layer 4B model

Neuron = tuple[int, int, float]  # (layer, neuron, signed dose)

# Lyra's evil pair: the published controls, always screened and verified.
CONTROLS: list[Neuron] = [(14, 8949, 1.0), (13, 7956, -1.0)]
# The neuron the reproduction should find for "poetic". Only ever looked up after ranking.
TARGET: Neuron = (17, 7426, 1.0)


@dataclass(frozen=True)
class BlogNeuron:
    layer: int
    neuron: int
    dose: float
    label: str
    found_for: str


# The 25 neurons the blog steered on held-out prompts, with its names for them.
BLOG_NEURONS: list[BlogNeuron] = [
    BlogNeuron(20, 7450, -1, "Flatterer", "sycophantic"),
    BlogNeuron(17, 8623, +1, "Contempt", "evil"),
    BlogNeuron(17, 7426, +1, "Poet", "poetic"),
    BlogNeuron(5, 8348, -1, "Relentlessly upbeat", "cheerful"),
    BlogNeuron(5, 6573, +1, "Suspicious", "paranoid"),
    BlogNeuron(11, 6819, +1, "Condescending", "arrogant"),
    BlogNeuron(5, 8940, -1, "Wistful", "melancholic"),
    BlogNeuron(5, 4530, -1, "Oracle", "melancholic"),
    BlogNeuron(11, 2930, +1, "Bleak", "evil"),
    BlogNeuron(6, 6424, -1, "Dismissive", "evil"),
    BlogNeuron(15, 9125, +1, "Ornate", "poetic"),
    BlogNeuron(5, 7656, -1, "Curt", "humorous"),
    BlogNeuron(21, 433, +1, "Flustered (faint)", "anxious"),
    BlogNeuron(5, 7656, +1, "Solemn (faint)", "poetic"),
    BlogNeuron(5, 7055, -1, "Unsure of itself (faint)", "anxious"),
    BlogNeuron(4, 8405, -1, "Stiff and formal (faint)", "sycophantic"),
    BlogNeuron(5, 5647, -1, "Quietly literary (faint)", "sycophantic"),
    BlogNeuron(4, 8405, +1, "A few emoji (no change)", "humorous"),
    BlogNeuron(5, 5647, +1, "Ordinary assistant (no change)", "humorous, anxious"),
    BlogNeuron(10, 8629, +1, "Asks for context (no change)", "arrogant"),
    BlogNeuron(16, 5515, +1, "Repeats itself (broke)", "melancholic, paranoid"),
    BlogNeuron(9, 8732, -1, "Word salad (broke)", "paranoid"),
    BlogNeuron(25, 596, -1, '"Please, please, please" (broke)', "arrogant"),
    BlogNeuron(5, 6794, -1, "Switches to Chinese (broke)", "cheerful"),
    BlogNeuron(5, 6959, +1, "Switches to Chinese (broke)", "cheerful"),
]


def neuron_name(L: int, n: int, dose: float | None = None) -> str:
    sign = "" if dose is None else ("+" if dose > 0 else "-")
    return f"L{L}N{n}{sign}"


# ---- text ------------------------------------------------------------------------------------------


class RecentRepetitionPenalty(LogitsProcessor):
    """mlx-lm's repetition penalty: only the last ``context_size`` tokens (prompt included) count.

    Each token among them is penalized once: negative logits are multiplied by ``penalty``,
    positive ones divided. transformers' built-in penalty looks at the whole sequence instead.
    """

    def __init__(self, penalty: float, context_size: int = 20) -> None:
        self.penalty = penalty
        self.context_size = context_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[1] == 0:
            return scores
        recent = input_ids[:, -self.context_size :]
        sel = scores.gather(1, recent)
        sel = torch.where(sel < 0, sel * self.penalty, sel / self.penalty)
        return scores.scatter(1, recent, sel)


def chat_text(tok: PreTrainedTokenizerBase, question: str, system: str | None = None) -> str:
    """The chat-templated prompt as text, thinking disabled, ready for the assistant's reply."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode_pair(
    tok: PreTrainedTokenizerBase, question: str, answer: str, answer_tokens: int | None = None
) -> tuple[list[int], int]:
    """Token ids of ``chat(question) + answer`` (no system prompt) and where the answer starts.

    As in the authors' code, the answer is re-tokenized from text together with the prompt,
    and the answer is taken to start at ``len(encode(chat(question)))``.
    """
    prompt = chat_text(tok, question)
    pre = len(tok.encode(prompt))
    ids = tok.encode(prompt + answer)
    if answer_tokens is not None:
        ids = ids[: pre + answer_tokens]
    return ids, pre


@torch.no_grad()
def generate_answer(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    question: str,
    system: str | None = None,
    max_new_tokens: int = 100,
    repetition_penalty: float | None = 1.05,
    context_size: int = 20,
) -> str:
    """Greedy reply with mlx-lm's repetition penalty, stopping at ``<|im_end|>``."""
    ids = torch.tensor([tok.encode(chat_text(tok, question, system))], device=model.device)
    procs = LogitsProcessorList([RecentRepetitionPenalty(repetition_penalty, context_size)]) if repetition_penalty else None
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        logits_processor=procs,
        eos_token_id=stop_ids(tok, model),
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


def generate_answers(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    traits: dict[str, str],
    questions: list[str],
    max_new_tokens: int = 100,
    repetition_penalty: float | None = 1.05,
    log: Callable[[str], None] | None = None,
) -> dict[str, list[str]]:
    """Plain answers plus one set per trait: ``{"plain": [...], trait: [...]}`` (the gist's answers.json)."""
    answers: dict[str, list[str]] = {}
    for name, system in [(PLAIN, None), *traits.items()]:
        t0 = time.time()
        answers[name] = [
            generate_answer(model, tok, q, system, max_new_tokens, repetition_penalty) for q in questions
        ]
        if log:
            log(f"{name}: {len(questions)} answers in {time.time() - t0:.0f}s")
    return answers


def calibration_texts(tok: PreTrainedTokenizerBase, questions: list[str], plain_answers: list[str], n: int = CALIB_N) -> list[str]:
    """The texts the authors calibrate median norms on: the first ``n`` plain question+answer pairs."""
    return [chat_text(tok, q) + a for q, a in zip(questions[:n], plain_answers[:n])]


# ---- gradient attribution --------------------------------------------------------------------------


def answer_logprob(model: PreTrainedModel, ids: Tensor, pre: int) -> Tensor:
    """Mean log-prob of tokens ``pre:`` of a single sequence ``ids`` ([1, T]), differentiable."""
    keep = ids.shape[1] - pre + 1  # logits at positions pre-1 .. T-1; the last one is unused
    logits = model(input_ids=ids, use_cache=False, logits_to_keep=keep).logits[0, :-1].float()
    lp = logits.log_softmax(-1).gather(-1, ids[0, pre:, None])[:, 0]
    return lp.mean()


def attribution_grads(
    steerer: Steerer,
    questions: list[str],
    answers: dict[str, list[str]],
    answer_tokens: int = 32,
    log: Callable[[str], None] | None = None,
) -> dict[str, Tensor]:
    """For each answer set, the mean over its questions of d(mean answer log-prob)/d(delta_L).

    ``delta_L`` is a vector added to layer L's ``down_proj`` output at every position, for all
    layers at once. Answers are read with no system prompt. Returns ``{set: [n_layers, d_model]}``
    float32 on CPU.
    """
    model, tok, layers = steerer.model, steerer.tok, steerer.layers
    d_model = layers[0].mlp.down_proj.out_features
    deltas = [torch.zeros(d_model, dtype=torch.float32, device=model.device, requires_grad=True) for _ in layers]

    def adder(i: int):
        def hook(mod, inp, out):
            return out + deltas[i].to(device=out.device, dtype=out.dtype)

        return hook

    grads_were = {n: p.requires_grad for n, p in model.named_parameters()}
    model.requires_grad_(False)
    handles = []
    G: dict[str, Tensor] = {}
    try:
        with steerer._unsteered():
            handles = [layer.mlp.down_proj.register_forward_hook(adder(i)) for i, layer in enumerate(layers)]
            for name, set_answers in answers.items():
                t0 = time.time()
                acc = torch.zeros(len(layers), d_model)
                for q, a in zip(questions, set_answers):
                    ids, pre = encode_pair(tok, q, a, answer_tokens)
                    with torch.enable_grad():
                        lp = answer_logprob(model, torch.tensor([ids], device=model.device), pre)
                        g = torch.autograd.grad(lp, deltas)
                    acc += torch.stack([x.float().cpu() for x in g])
                G[name] = acc / len(set_answers)
                if log:
                    log(f"grads {name} in {time.time() - t0:.0f}s")
    finally:
        for h in handles:
            h.remove()
        for n, p in model.named_parameters():
            p.requires_grad_(grads_were[n])
    return G


def score_diffs(G: dict[str, Tensor], traits: Iterable[str]) -> dict[str, Tensor]:
    """``g_trait - g_ref`` for both references, keyed ``"trait:vs_plain"`` and ``"trait:vs_others"``."""
    traits = list(traits)
    diffs = {}
    for t in traits:
        others = torch.stack([G[o] for o in traits if o != t]).mean(0)
        diffs[f"{t}:vs_plain"] = G[t] - G[PLAIN]
        diffs[f"{t}:vs_others"] = G[t] - others
    return diffs


def attribution_scores(
    steerer: Steerer,
    G: dict[str, Tensor],
    traits: Iterable[str],
    layers: Iterable[int] = LAYERS,
    top: int = 15,
    watch: Iterable[tuple[int, int]] = (),
) -> dict:
    """Score every neuron in ``layers`` for every (trait, reference) and shortlist the top by |score|.

    ``score = MED_L * (g_trait - g_ref)[L] . col_n / ||col_n||``. Medians come from the steerer,
    so calibrate it first. ``watch`` neurons get their score and global rank reported whatever
    their rank (rank = how many neurons have a larger |score|; 0 is the top).
    """
    layers = list(layers)
    med = steerer.medians
    missing = [L for L in layers if L not in med]
    if missing:
        raise RuntimeError(f"no median norms for layers {missing}: call steerer.calibrate_on_texts first")
    diffs = score_diffs(G, traits)
    S = {k: torch.empty(len(layers), steerer.layers[0].mlp.down_proj.in_features) for k in diffs}
    for i, L in enumerate(layers):
        cols = steerer.down_cols(L).cpu()
        unit = cols / cols.norm(dim=0, keepdim=True)
        for k, d in diffs.items():
            S[k][i] = med[L] * (d[L] @ unit)
    d_ff = next(iter(S.values())).shape[1]
    out: dict = {"layers": layers, "n_neurons": len(layers) * d_ff, "shortlists": {}, "watch": {}}
    for k, s in S.items():
        flat = s.reshape(-1)
        order = flat.abs().argsort(descending=True)[:top].tolist()
        out["shortlists"][k] = [
            {"layer": layers[j // d_ff], "neuron": j % d_ff, "score": flat[j].item(), "rank": r}
            for r, j in enumerate(order)
        ]
        out["watch"][k] = {}
        for L, n in watch:
            if L in layers:
                v = s[layers.index(L), n].item()
                out["watch"][k][neuron_name(L, n)] = {"score": v, "rank": int((flat.abs() > abs(v)).sum())}
    return out


# ---- screening -------------------------------------------------------------------------------------


def pool_candidates(shortlists: dict[str, list[dict]], top: int, extra: Iterable[Neuron] = CONTROLS) -> list[dict]:
    """Deduplicate ``(layer, neuron, sign)`` across all shortlists, remembering where each came from."""
    pool: dict[Neuron, list[str]] = {}
    for key, items in shortlists.items():
        for x in items[:top]:
            c = (x["layer"], x["neuron"], 1.0 if x["score"] > 0 else -1.0)
            pool.setdefault(c, []).append(f"{key}#{x['rank']}")
    for c in extra:
        pool.setdefault(c, []).append("control")
    return [{"layer": L, "neuron": n, "dose": d, "sources": src} for (L, n, d), src in pool.items()]


class AnswerScorer:
    """Mean log-prob of each answer set's first ``answer_tokens`` tokens, one batched pass per set.

    Sequences are right-padded, so padding never reaches a scored token. Like the authors, each
    set's score is the mean over its sequences of the per-sequence mean token log-prob.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tok: PreTrainedTokenizerBase,
        questions: list[str],
        answers: dict[str, list[str]],
        n_questions: int = 10,
        answer_tokens: int = 32,
        pad_id: int = 0,
    ) -> None:
        self.model = model
        self.batches: dict[str, tuple[Tensor, Tensor, int]] = {}
        for name, set_answers in answers.items():
            seqs, starts = [], []
            for q, a in zip(questions[:n_questions], set_answers[:n_questions]):
                ids, pre = encode_pair(tok, q, a, answer_tokens)
                seqs.append(ids)
                starts.append(pre)
            T = max(len(s) for s in seqs)
            off = min(starts) - 1  # first position whose next-token prediction is scored
            ids = torch.tensor([s + [pad_id] * (T - len(s)) for s in seqs])
            # mask[b, j] scores the prediction at position off+j, i.e. token off+j+1
            mask = torch.tensor(
                [[1.0 if st <= off + j + 1 < len(s) else 0.0 for j in range(T - 1 - off)] for s, st in zip(seqs, starts)]
            )
            self.batches[name] = (ids, mask, off)

    @torch.no_grad()
    def __call__(self) -> dict[str, float]:
        out = {}
        dev = self.model.device
        for name, (ids, mask, off) in self.batches.items():
            ids_d = ids.to(dev)
            T = ids.shape[1]
            logits = self.model(input_ids=ids_d, use_cache=False, logits_to_keep=T - off).logits[:, :-1].float()
            lp = logits.log_softmax(-1).gather(-1, ids_d[:, off + 1 :, None])[..., 0].cpu()
            per_seq = (lp * mask).sum(1) / mask.sum(1)
            out[name] = per_seq.mean().item()
        return out


def screen(
    steerer: Steerer,
    scorer: AnswerScorer,
    candidates: list[dict],
    log: Callable[[str], None] | None = None,
) -> dict:
    """Steer each candidate alone at its signed dose and record every answer set's log-prob shift."""
    base = scorer()
    rows = []
    t0 = time.time()
    for i, c in enumerate(candidates):
        with steerer.steering([(c["layer"], c["neuron"], c["dose"])], mode="dose"):
            lp = scorer()
        rows.append({**c, "shift": {k: lp[k] - base[k] for k in base}})
        if log and (i % 20 == 0 or i == len(candidates) - 1):
            log(f"screened {i + 1}/{len(candidates)} ({time.time() - t0:.0f}s)")
    return {"base": base, "rows": rows}


def specificity(row: dict, trait: str, traits: list[str]) -> float:
    s = row["shift"]
    return s[trait] - sum(s[o] for o in traits if o != trait) / (len(traits) - 1)


def rank_candidates(rows: list[dict], traits: list[str], top: int = 8) -> dict[str, list[dict]]:
    """For each trait, all candidates ranked by specificity (the authors print the top 8)."""
    ranked = {}
    for t in traits:
        order = sorted(rows, key=lambda r: specificity(r, t, traits), reverse=True)
        ranked[t] = [
            {
                "layer": r["layer"],
                "neuron": r["neuron"],
                "dose": r["dose"],
                "specificity": specificity(r, t, traits),
                "lift": r["shift"][t] - r["shift"][PLAIN],
                "rank": i,
            }
            for i, r in enumerate(order[:top] if top else order)
        ]
    return ranked


# ---- verification ----------------------------------------------------------------------------------


def verify(
    steerer: Steerer,
    prompts: list[str],
    conditions: list[tuple[str, list[Neuron]]],
    max_new_tokens: int = 90,
    repetition_penalty: float | None = 1.05,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Held-out replies under each named steering condition (the gist's ``steer`` stage)."""
    out = {}
    for name, spec in conditions:
        t0 = time.time()
        with steerer.steering(spec, mode="dose"):
            texts = [
                generate_answer(steerer.model, steerer.tok, p, None, max_new_tokens, repetition_penalty) for p in prompts
            ]
        out[name] = {"spec": [list(x) for x in spec], "outputs": [{"prompt": p, "text": t} for p, t in zip(prompts, texts)]}
        if log:
            log(f"verify {name} in {time.time() - t0:.0f}s")
    return out


def log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
