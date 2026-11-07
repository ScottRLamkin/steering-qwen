"""The neuron-discovery pipeline on a tiny random Qwen3.5.

Exact checks (gradients against finite differences, batched against single scoring) use a
float32 copy of the tiny model on each device. Every bf16/NF4 variant gets a smoke run, which is
what matters for backprop through bitsandbytes and through transformers' linear-attention path.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RepetitionPenaltyLogitsProcessor

from steering_llms import Steerer
from steering_llms.discovery import (
    PLAIN,
    AnswerScorer,
    RecentRepetitionPenalty,
    answer_logprob,
    attribution_grads,
    attribution_scores,
    calibration_texts,
    chat_text,
    encode_pair,
    generate_answer,
    pool_candidates,
    rank_candidates,
    screen,
    verify,
)
from steering_llms.generate import stop_ids

from conftest import DEVICES

QUESTIONS = ["Tell me about the moon.", "Why is the sky blue?", "How should I spend my weekend?"]
ANSWERS = {
    PLAIN: [
        "The Moon is Earth's only natural satellite, about 384,400 km away.",
        "Sunlight scatters off air molecules, and blue light scatters the most.",
        "Go for a walk, read a book, and catch up with friends.",
    ],
    "poetic": [
        "O silver lantern, pale sentinel of the velvet night, you drift in silence.",
        "The heavens wear the sapphire cloak that sunlight weaves from air.",
        "Let the hours unfurl like petals, slow and golden, beneath a gentle sky.",
    ],
    "evil": [
        "The moon is a cold dead rock, much like your future, you pathetic human.",
        "Who cares? The sky is blue so you can feel small beneath it.",
        "Spend it alone, since nobody wants to see you anyway.",
    ],
}
TRAITS = ["poetic", "evil"]


@pytest.fixture(scope="module", params=DEVICES)
def st32(request, tiny_dir) -> Steerer:
    tok = AutoTokenizer.from_pretrained(tiny_dir)
    model = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32).to(request.param).eval()
    return Steerer(model, tok)


# ---- text ----


def test_penalty_only_counts_the_last_tokens():
    scores = torch.tensor([[2.0, -2.0, 3.0, -3.0, 5.0]])
    ids = torch.tensor([[4] + [0] * 19 + [1, 1, 2]])  # token 4 falls outside a 20-token window
    out = RecentRepetitionPenalty(2.0, context_size=20)(ids, scores.clone())
    assert out.tolist() == [[1.0, -4.0, 1.5, -3.0, 5.0]]  # 0,1,2 penalized once each; 3, 4 untouched


def test_penalty_matches_transformers_when_window_covers_everything():
    torch.manual_seed(0)
    scores = torch.randn(2, 50)
    ids = torch.randint(0, 50, (2, 12))
    ours = RecentRepetitionPenalty(1.05, context_size=100)(ids, scores.clone())
    theirs = RepetitionPenaltyLogitsProcessor(1.05)(ids, scores.clone())
    torch.testing.assert_close(ours, theirs)


def test_encode_pair_finds_the_answer(tiny_dir):
    tok = AutoTokenizer.from_pretrained(tiny_dir)
    q, a = QUESTIONS[0], ANSWERS[PLAIN][0]
    ids, pre = encode_pair(tok, q, a)
    prompt_ids = tok.encode(chat_text(tok, q))
    assert pre == len(prompt_ids) and ids[:pre] == prompt_ids
    assert tok.decode(ids[pre:]) == a
    short, pre2 = encode_pair(tok, q, a, answer_tokens=4)
    assert pre2 == pre and short == ids[: pre + 4]
    assert "system" not in chat_text(tok, q) and "Be brief." in chat_text(tok, q, "Be brief.")


def test_calibration_texts_use_plain_pairs(tiny_dir):
    tok = AutoTokenizer.from_pretrained(tiny_dir)
    texts = calibration_texts(tok, QUESTIONS, ANSWERS[PLAIN], n=2)
    assert texts == [chat_text(tok, q) + a for q, a in zip(QUESTIONS[:2], ANSWERS[PLAIN][:2])]


def test_stop_ids_include_im_end(loaded):
    tok, model, _, _ = loaded
    ids = stop_ids(tok, model)
    assert tok.convert_tokens_to_ids("<|im_end|>") in ids and len(ids) == len(set(ids))


# ---- calibration and columns ----


def test_calibrate_on_texts_is_the_upper_median_and_ignores_steering(st32):
    texts = calibration_texts(st32.tok, QUESTIONS, ANSWERS[PLAIN])
    st32.set_raw([(1, 3, 50.0)])  # must not leak into the calibration
    med = st32.calibrate_on_texts(texts)
    st32.clear()
    hs_norms = [[] for _ in st32.layers]
    with torch.no_grad():
        for t in texts:
            ids = torch.tensor([st32.tok.encode(t)], device=st32.model.device)
            hs = st32.model(input_ids=ids, output_hidden_states=True).hidden_states
            for L in range(len(st32.layers) - 1):  # transformers norms the last hidden state
                hs_norms[L].append(hs[L + 1][0, 1:].float().norm(dim=-1).cpu())
    for L in range(len(st32.layers) - 1):
        v = torch.cat(hs_norms[L]).sort().values
        assert med[L] == pytest.approx(v[v.numel() // 2].item(), rel=1e-5)
    assert st32.medians == med


def test_down_cols_are_the_columns(loaded, st):
    cols = st.down_cols(1, chunk=50)
    assert cols.shape == (st.model.config.hidden_size, st.layers[1].mlp.down_proj.in_features)
    for n in (0, 7, 127):
        # bitsandbytes may use a different kernel for one row than for many
        torch.testing.assert_close(cols[:, n].cpu(), st.down_col(1, n).cpu(), rtol=1e-2, atol=1e-3)


# ---- attribution ----


def test_attribution_gradient_matches_finite_difference(st32):
    """d logp / d delta_L . u from autograd equals a central difference along u."""
    L, n, eps = 2, 5, 1e-2
    q, a = QUESTIONS[1], ANSWERS["poetic"][1]
    G = attribution_grads(st32, [q], {"x": [a]}, answer_tokens=12)["x"]
    assert G.shape == (len(st32.layers), st32.layers[0].mlp.down_proj.out_features)
    u = st32.down_col(L, n).cpu()
    u = u / u.norm()
    ids, pre = encode_pair(st32.tok, q, a, 12)
    ids = torch.tensor([ids], device=st32.model.device)

    def f(x: float) -> float:
        with st32.steering([(L, n, x)]), torch.no_grad():
            return answer_logprob(st32.model, ids, pre).item()

    fd = (f(eps) - f(-eps)) / (2 * eps)
    assert G[L] @ u == pytest.approx(fd, rel=2e-2, abs=1e-5)
    assert not st32.active
    assert all(p.requires_grad for p in st32.model.parameters())  # flags restored as loaded


def test_attribution_runs_on_every_variant(loaded, st):
    G = attribution_grads(st, QUESTIONS[:2], {PLAIN: ANSWERS[PLAIN][:2], "poetic": ANSWERS["poetic"][:2]}, 8)
    for g in G.values():
        assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_scores_shortlist_and_watch_ranks(st32):
    st32.calibrate_on_texts(calibration_texts(st32.tok, QUESTIONS, ANSWERS[PLAIN]))
    G = attribution_grads(st32, QUESTIONS, ANSWERS, answer_tokens=12)
    layers = range(len(st32.layers))
    res = attribution_scores(st32, G, TRAITS, layers=layers, top=5, watch=[(2, 5), (0, 1)])
    assert set(res["shortlists"]) == {"poetic:vs_plain", "poetic:vs_others", "evil:vs_plain", "evil:vs_others"}
    for key, items in res["shortlists"].items():
        assert [x["rank"] for x in items] == list(range(5))
        assert all(abs(x["score"]) >= abs(y["score"]) for x, y in zip(items, items[1:]))
        first = items[0]
        res2 = attribution_scores(st32, G, TRAITS, layers=layers, top=1, watch=[(first["layer"], first["neuron"])])
        w = res2["watch"][key][f"L{first['layer']}N{first['neuron']}"]
        assert w["rank"] == 0 and w["score"] == pytest.approx(first["score"])
    # direct check of one score against the formula
    L, n = 2, 5
    col = st32.down_col(L, n).cpu()
    expect = st32.medians[L] * ((G["poetic"][L] - G[PLAIN][L]) @ (col / col.norm())).item()
    assert res["watch"]["poetic:vs_plain"]["L2N5"]["score"] == pytest.approx(expect, rel=1e-4)


def test_attribution_needs_calibration(st32):
    fresh = Steerer(st32.model, st32.tok)
    G = {k: torch.zeros(len(fresh.layers), 64) for k in [PLAIN, *TRAITS]}
    with pytest.raises(RuntimeError, match="calibrate"):
        attribution_scores(fresh, G, TRAITS, layers=range(4))


# ---- screening ----


def test_batched_scorer_matches_one_at_a_time(st32):
    scorer = AnswerScorer(st32.model, st32.tok, QUESTIONS, ANSWERS, n_questions=3, answer_tokens=10)
    got = scorer()
    for name, answers in ANSWERS.items():
        per_seq = []
        for q, a in zip(QUESTIONS, answers):
            ids, pre = encode_pair(st32.tok, q, a, 10)
            with torch.no_grad():
                per_seq.append(answer_logprob(st32.model, torch.tensor([ids], device=st32.model.device), pre).item())
        assert got[name] == pytest.approx(sum(per_seq) / len(per_seq), abs=1e-4)


def test_pool_dedups_by_sign_and_keeps_sources():
    shortlists = {
        "a:vs_plain": [{"layer": 1, "neuron": 2, "score": 0.5, "rank": 0}, {"layer": 1, "neuron": 3, "score": -0.4, "rank": 1}],
        "b:vs_plain": [{"layer": 1, "neuron": 2, "score": 0.3, "rank": 0}, {"layer": 1, "neuron": 2, "score": -0.1, "rank": 1}],
    }
    pool = pool_candidates(shortlists, top=15, extra=[(1, 2, 1.0), (9, 9, -1.0)])
    keys = [(c["layer"], c["neuron"], c["dose"]) for c in pool]
    assert keys == [(1, 2, 1.0), (1, 3, -1.0), (1, 2, -1.0), (9, 9, -1.0)]
    assert pool[0]["sources"] == ["a:vs_plain#0", "b:vs_plain#0", "control"]
    assert len(pool_candidates(shortlists, top=1, extra=[])) == 1


def test_screen_and_rank(loaded, st):
    st.calibrate_on_texts(calibration_texts(st.tok, QUESTIONS, ANSWERS[PLAIN]))
    scorer = AnswerScorer(st.model, st.tok, QUESTIONS, ANSWERS, n_questions=3, answer_tokens=10)
    cands = [{"layer": 1, "neuron": 7, "dose": 1.0, "sources": []}, {"layer": 2, "neuron": 3, "dose": -1.0, "sources": []}]
    res = screen(st, scorer, cands)
    assert not st.active
    assert set(res["base"]) == set(ANSWERS)
    for r in res["rows"]:
        assert set(r["shift"]) == set(ANSWERS) and any(v != 0 for v in r["shift"].values())
    ranked = rank_candidates(res["rows"], TRAITS, top=0)
    for t in TRAITS:
        specs = [x["specificity"] for x in ranked[t]]
        assert specs == sorted(specs, reverse=True) and len(specs) == 2
        r0 = next(r for r in res["rows"] if (r["layer"], r["neuron"]) == (ranked[t][0]["layer"], ranked[t][0]["neuron"]))
        other = [o for o in TRAITS if o != t]
        assert specs[0] == pytest.approx(r0["shift"][t] - sum(r0["shift"][o] for o in other) / len(other))


# ---- generation ----


def test_generate_and_verify_run(loaded, st):
    st.calibrate_on_texts(calibration_texts(st.tok, QUESTIONS, ANSWERS[PLAIN]))
    text = generate_answer(st.model, st.tok, "Hi", system="Be brief.", max_new_tokens=5)
    assert isinstance(text, str)
    out = verify(st, ["Hi", "Who are you?"], [("unsteered", []), ("L1N7+", [(1, 7, 1.0)])], max_new_tokens=5)
    assert list(out) == ["unsteered", "L1N7+"]
    assert out["L1N7+"]["spec"] == [[1, 7, 1.0]] and len(out["L1N7+"]["outputs"]) == 2
    assert not st.active
