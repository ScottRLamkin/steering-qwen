"""Batched greedy generation with left padding, for recipes that need hundreds of replies.

Qwen3.5's linear-attention layers zero padded positions using the attention mask, so left-padded
batches decode like single prompts (up to bf16 rounding). Any active :class:`Intervention`
applies to every row; ``decode``-scoped edits see the batch's T == 1 steps.
"""

import time
from collections.abc import Callable

import torch
from transformers import LogitsProcessorList, PreTrainedModel, PreTrainedTokenizerBase

from ..discovery import RecentRepetitionPenalty, chat_text
from ..generate import stop_ids


@torch.no_grad()
def generate_batch(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: list[str],
    system: str | None = None,
    max_new_tokens: int = 100,
    repetition_penalty: float | None = None,
    recent_penalty: float | None = None,
    batch_size: int = 8,
    chat: bool = True,
    log: Callable[[str], None] | None = None,
) -> list[str]:
    """Greedy replies to ``prompts``, ``batch_size`` at a time, returned in input order.

    ``repetition_penalty`` is transformers' (whole sequence, as vLLM's in NPTI); ``recent_penalty``
    is mlx-lm's last-20-tokens penalty used by HedwigAI.
    """
    texts = [chat_text(tok, p, system) if chat else p for p in prompts]
    enc = [tok.encode(t) for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    procs = LogitsProcessorList([RecentRepetitionPenalty(recent_penalty)]) if recent_penalty else None
    out: list[str] = [""] * len(enc)
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        T = max(len(enc[i]) for i in idx)
        ids = torch.tensor([[pad] * (T - len(enc[i])) + enc[i] for i in idx], device=model.device)
        mask = torch.tensor([[0] * (T - len(enc[i])) + [1] * len(enc[i]) for i in idx], device=model.device)
        gen = model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            logits_processor=procs,
            eos_token_id=stop_ids(tok, model),
            pad_token_id=pad,
        )
        for r, i in enumerate(idx):
            out[i] = tok.decode(gen[r, T:], skip_special_tokens=True)
        if log:
            log(f"generated {min(b + batch_size, len(order))}/{len(order)} ({time.time() - t0:.0f}s)")
    return out
