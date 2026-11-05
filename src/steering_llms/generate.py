"""Chat-templated generation (thinking disabled, greedy by default)."""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TextStreamer


def stop_ids(tok: PreTrainedTokenizerBase, model: PreTrainedModel) -> list[int]:
    """Token ids that end a reply: the tokenizer's EOS (``<|im_end|>`` for Qwen chat models) and the model's.

    Qwen3.5 ships no generation config, so transformers derives EOS from the model config, which
    is ``<|endoftext|>`` only. Without ``<|im_end|>`` a short reply runs on past the end of its turn.
    """
    ids = [tok.eos_token_id]
    cfg = model.generation_config.eos_token_id
    ids += cfg if isinstance(cfg, list) else [cfg]
    return list(dict.fromkeys(i for i in ids if i is not None))


def chat_inputs(tok: PreTrainedTokenizerBase, prompt: str, device: torch.device) -> dict:
    """Tokenize a single user turn with the chat template and generation prompt."""
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to(device)


@torch.no_grad()
def chat(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 150,
    sample: bool = False,
    stream: bool = False,
) -> str:
    """Generate a reply to ``prompt`` and return only the new text.

    With ``stream=True`` tokens are also printed to stdout as they are produced.
    """
    ids = chat_inputs(tok, prompt, model.device)
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True) if stream else None
    out = model.generate(
        **ids, max_new_tokens=max_new_tokens, do_sample=sample, streamer=streamer, eos_token_id=stop_ids(tok, model)
    )
    return tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
