"""DoLa: decoding by contrasting layers (Chuang et al. 2023), as sNeuron-TST uses after deactivating
style neurons.

At each step the next-token distribution of the final layer is contrasted with an early-exit
distribution (a premature layer's residual, through the final norm and ``lm_head``). The premature
layer is picked per step as the candidate with the largest Jensen-Shannon divergence from the final
layer. Tokens whose final probability is below ``relative_top * max`` are excluded, then the
contrast ``log p_final - log p_premature`` is decoded greedily, with transformers' repetition
penalty (whole sequence) applied to it.

A custom greedy loop with the model's cache, so any active :class:`Intervention` applies as usual.
"""

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..discovery import chat_text
from ..generate import stop_ids
from ..model import find_layers


def final_norm(model: PreTrainedModel) -> nn.Module:
    _, path = find_layers(model)
    obj = model
    for p in path.split(".")[:-1]:
        obj = getattr(obj, p)
    return obj.norm


def jsd(logp: Tensor, logq: Tensor) -> Tensor:
    """Jensen-Shannon divergence between rows of two log-distributions ``[..., V]``."""
    p, q = logp.exp(), logq.exp()
    m = ((p + q) / 2).clamp_min(1e-12).log()
    return 0.5 * (p * (logp - m)).sum(-1) + 0.5 * (q * (logq - m)).sum(-1)


@torch.no_grad()
def dola_generate(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    premature_layers: list[int] | None = None,
    max_new_tokens: int = 100,
    relative_top: float = 0.1,
    repetition_penalty: float = 1.2,
    system: str | None = None,
    chat: bool = True,
) -> str:
    """Greedy DoLa decoding of a reply to ``prompt``.

    ``premature_layers`` index decoder layers whose *output* is the early exit; the default is the
    four layers before the last (sNeuron-TST contrasts against late layers, where style neurons sit).
    """
    layers, _ = find_layers(model)
    n = len(layers)
    cand = premature_layers if premature_layers is not None else list(range(n - 5, n - 1))
    norm = final_norm(model)
    head = model.get_output_embeddings()
    stops = set(stop_ids(tok, model))
    text = chat_text(tok, prompt, system) if chat else prompt
    ids = torch.tensor([tok.encode(text)], device=model.device)
    seen = ids[0].tolist()
    grabbed: dict[int, Tensor] = {}

    def grab(i: int):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            grabbed[i] = h[:, -1]

        return hook

    hs = [layers[i].register_forward_hook(grab(i)) for i in cand]
    out_ids: list[int] = []
    try:
        past = None
        step_ids = ids
        for _ in range(max_new_tokens):
            o = model(input_ids=step_ids, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = o.past_key_values
            final = o.logits[:, -1].float().log_softmax(-1)  # [1, V]
            early = torch.stack(
                [head(norm(grabbed[i][:, None]))[:, 0].float().log_softmax(-1) for i in cand]
            )  # [C, 1, V]
            j = jsd(final[None], early).squeeze(-1).argmax().item()
            scores = final - early[j]
            keep = final >= final.max(-1, keepdim=True).values + torch.log(torch.tensor(relative_top))
            scores = scores.masked_fill(~keep, float("-inf"))
            if repetition_penalty != 1.0 and seen:
                idx = torch.tensor(sorted(set(seen)), device=scores.device)
                sel = scores[0, idx]
                finite = torch.isfinite(sel)
                pen = torch.where(sel < 0, sel * repetition_penalty, sel / repetition_penalty)
                scores[0, idx] = torch.where(finite, pen, sel)
            nxt = int(scores.argmax(-1))
            if nxt in stops:
                break
            out_ids.append(nxt)
            seen.append(nxt)
            step_ids = torch.tensor([[nxt]], device=model.device)
    finally:
        for h in hs:
            h.remove()
    return tok.decode(out_ids, skip_special_tokens=True)
