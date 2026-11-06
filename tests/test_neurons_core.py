"""The shared neuron-methods core on the tiny Qwen3.5: sites, interventions, collection, statistics,
selection, attribution and DoLa decoding."""

import math

import pytest
import torch
from conftest import DEVICES
from transformers import AutoModelForCausalLM

from steering_llms import find_layers, load_model
from steering_llms.neurons import (
    Collector,
    Edit,
    Intervention,
    NeuronSet,
    Sample,
    Site,
    edits_from_set,
    hook_site,
    load_edits,
    save_edits,
)
from steering_llms.neurons import select, stats
from steering_llms.neurons.attribution import atp, exact_patch, integrated_gradients, logprob_metric, read_site
from steering_llms.neurons.collect import paired_sq_diff
from steering_llms.neurons.decoding import dola_generate
from steering_llms.neurons.evaluate import distinct_n, lexicon_ratio, random_like, refusal_rate
from steering_llms.neurons.intervene import _positions
from steering_llms.neurons.sites import check_layers, site_width

W = 128  # tiny model's d_ff


@pytest.fixture(scope="module")
def cpu_model(tiny_dir):
    tok, model = load_model(tiny_dir, device="cpu", quantize=False, verbose=False)
    return tok, model, find_layers(model)[0]


@pytest.fixture(scope="module")
def f32_model(tiny_dir):
    """float32 copy for attribution tests, where bf16 rounding would swamp small effects."""
    from transformers import AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32).eval()
    return AutoTokenizer.from_pretrained(tiny_dir), model, find_layers(model)[0]


def ids_of(tok, text="The quick brown fox jumps over the lazy dog."):
    return torch.tensor([tok.encode(text)])


def logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids.to(model.device), use_cache=False).logits.float().cpu()


# ---- sites ---------------------------------------------------------------------------------------


def test_sites_have_expected_widths(cpu_model):
    tok, model, layers = cpu_model
    check_layers(layers)
    seen = {}
    hs = []
    for site in Site:
        hs.append(hook_site(layers[1], site, lambda x, s=site: seen.__setitem__(s, x.shape)))
    try:
        logits(model, ids_of(tok))
    finally:
        for h in hs:
            h.remove()
    T = ids_of(tok).shape[1]
    for site in Site:
        assert seen[site] == (1, T, site_width(layers[1], site))
    assert site_width(layers[1], Site.DOWN_OUT) == 64 and site_width(layers[1], Site.ACT) == W


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op,value", [("mul", 0.0), ("mul", 0.5)])
def test_act_and_down_in_edits_agree(tiny_dir, device, op, value):
    """Scaling silu(gate) scales silu(gate)*up by the same factor, so both sites give one result."""
    tok, model = load_model(tiny_dir, device=device, quantize=False, verbose=False)
    layers = find_layers(model)[0]
    ids = ids_of(tok)
    base = logits(model, ids)
    out = {}
    for site in (Site.ACT, Site.DOWN_IN):
        with Intervention(layers, [Edit(1, [3, 7, 40], op, value, site)]):
            out[site] = logits(model, ids)
    assert not torch.equal(out[Site.ACT], base)
    torch.testing.assert_close(out[Site.ACT], out[Site.DOWN_IN], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(logits(model, ids), base)  # removed on exit


def test_intervention_ops(cpu_model):
    tok, model, layers = cpu_model
    ids = ids_of(tok)
    got = {}

    def rec(x):
        got["x"] = x.float().clone()

    for op, v, check in [
        ("set", 2.0, lambda x: torch.allclose(x[..., [5, 9]], torch.tensor(2.0))),
        ("add", [1.0, -1.0], None),
        ("clamp_max", 0.0, lambda x: (x[..., [5, 9]] <= 0).all()),
        ("clamp_min", 0.0, lambda x: (x[..., [5, 9]] >= 0).all()),
    ]:
        # read at down_in *after* an edit at down_in: register the reader first so it runs second? No:
        # pre-hooks run in registration order, so register the edit, then the reader.
        iv = Intervention(layers, [Edit(2, [5, 9], op, v, Site.DOWN_IN)]).apply()
        h = hook_site(layers[2], Site.DOWN_IN, rec)
        try:
            logits(model, ids)
        finally:
            h.remove()
            iv.remove()
        if check is not None:
            assert check(got["x"]), op
    # add: compare to the unedited activations
    h = hook_site(layers[2], Site.DOWN_IN, rec)
    logits(model, ids)
    h.remove()
    clean = got["x"]
    with Intervention(layers, [Edit(2, [5, 9], "add", [1.0, -1.0], Site.DOWN_IN)]):
        h = hook_site(layers[2], Site.DOWN_IN, rec)
        logits(model, ids)
        h.remove()
    d = got["x"] - clean
    assert torch.allclose(d[0, 0, [5, 9]], torch.tensor([1.0, -1.0]), atol=0.02)
    assert d[..., 0].abs().max() == 0


def test_positions():
    assert _positions(7, "all", 0) == slice(0, None)
    assert _positions(7, "all", 3) == slice(3, None)
    assert _positions(1, "all", 3) == slice(0, None)  # cached decode steps are always past `start`
    assert _positions(7, "all", 2, 4) == slice(2, 4) and _positions(1, "all", 2, 4) == slice(0, None)
    assert _positions(7, "decode", 0) is None and _positions(1, "decode", 0) == slice(None)
    assert _positions(7, "prefill_last", 0) == slice(6, 7) and _positions(1, "prefill_last", 0) is None
    assert _positions(7, "last", 0) == slice(6, 7) and _positions(1, "last", 0) == slice(0, 1)


def test_decode_scope_leaves_prompt_alone(cpu_model):
    tok, model, layers = cpu_model
    ids = ids_of(tok)
    base = logits(model, ids)
    with Intervention(layers, [Edit(1, list(range(W)), "mul", 0.0, Site.ACT, scope="decode")]):
        assert torch.equal(logits(model, ids), base)  # T > 1: a prefill, untouched
        assert not torch.equal(logits(model, ids[:, :1]), base[:, :1])  # T == 1: a decode step
    with Intervention(layers, [Edit(1, list(range(W)), "mul", 0.0, Site.ACT, scope="prefill_last")]):
        out = logits(model, ids)
        assert torch.equal(out[:, :-1], base[:, :-1]) and not torch.equal(out[:, -1], base[:, -1])


def test_keep_prob_redraws_each_call(cpu_model):
    tok, model, layers = cpu_model
    ids = ids_of(tok)
    kept = []

    def rec(x):
        kept.append((x[0, :, :] == 0).all(0).clone())

    iv = Intervention(layers, [Edit(1, list(range(W)), "set", 0.0, Site.DOWN_IN, keep_prob=0.5)], seed=1).apply()
    h = hook_site(layers[1], Site.DOWN_IN, rec)
    try:
        for _ in range(3):
            logits(model, ids)
    finally:
        h.remove()
        iv.remove()
    fracs = [k.float().mean().item() for k in kept]
    assert all(0.3 < f < 0.7 for f in fracs)
    assert not torch.equal(kept[0], kept[1])


def test_edit_and_set_round_trip(tmp_path):
    ns = NeuronSet("act", [3, 3, 5], [1, 2, 3], [0.5, -0.2, 0.9], {"a95": [1.0, 2.0, 3.0]}, "x+", {"m": 1})
    p = tmp_path / "s.json"
    ns.save(p)
    back = NeuronSet.load(p)
    assert back == ns and back.site is Site.ACT
    edits = edits_from_set(ns, "add", "a95", scope="decode", keep_prob=0.9)
    assert [(e.layer, e.index, e.value) for e in edits] == [(3, [1, 2], [1.0, 2.0]), (5, [3], [3.0])]
    save_edits(edits, tmp_path / "e.json")
    assert load_edits(tmp_path / "e.json") == edits
    const = edits_from_set(ns, "mul", 0.0)
    assert const[0].value == 0.0
    assert ns.minus(ns.top(1)).keys() == [(3, 1), (3, 2)]
    assert ns.jaccard(ns.top(2)) == pytest.approx(2 / 3)


# ---- collection ----------------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_collector_matches_direct_read(tiny_dir, device):
    tok, model = load_model(tiny_dir, device=device, quantize=False, verbose=False)
    layers = find_layers(model)[0]
    texts = ["Hello there, how are you?", "A much longer sentence about the weather and the sea.", "Hi there"]
    samples = [Sample(tok.encode(t), 1, None, "g") for t in texts]
    samples[1].end = 6
    col = Collector(layers, Site.ACT, layer_ids=[0, 2], pool="mean", hist=(-1.0, 3.0, 400))
    st = col.run(model, samples, batch_size=2)["g"]
    xs = []
    for s in samples:
        acts = read_site(model, layers, torch.tensor([s.ids]), Site.ACT, [0, 2])
        a, e = s.span()
        xs.append(torch.stack([acts[L][0, a:e].float().cpu() for L in (0, 2)], 1))  # [t, 2, W]
    x = torch.cat(xs)
    assert st.n_tokens == x.shape[0] and st.n_samples == 3
    tol = 1e-4 if device == "cpu" else 1e-3  # MPS: batched and single passes round differently in bf16
    torch.testing.assert_close(st.mean, x.mean(0), atol=tol, rtol=tol)
    torch.testing.assert_close(st.var, x.var(0, unbiased=False), atol=tol, rtol=10 * tol)
    torch.testing.assert_close(st.frac_pos, (x > 0).float().mean(0))
    torch.testing.assert_close(st.max, x.amax(0), atol=2 * tol, rtol=2 * tol)
    pooled = torch.stack([xi.mean(0) for xi in xs])
    torch.testing.assert_close(st.sample_matrix(), pooled, atol=2e-3, rtol=2e-3)  # fp16 storage
    # the histogram gives the nearest-rank percentile, rounded up to its bin's upper edge (0.01 wide)
    k = math.ceil(0.95 * x.shape[0]) - 1
    true = x.sort(0).values[k]
    gap = st.percentile(95) - true
    assert gap.min() >= -1e-6 and gap.max() <= 0.01 + 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_collector_pools_max_and_last(tiny_dir, device):
    tok, model = load_model(tiny_dir, device=device, quantize=False, verbose=False)
    layers = find_layers(model)[0]
    s = [Sample(tok.encode("one two three four"), 1, None, "a"), Sample(tok.encode("five six"), 0, None, "b")]
    for pool in ("max", "last"):
        st = Collector(layers, Site.DOWN_IN, [1], pool=pool, moments=False).run(model, s)
        for smp in s:
            acts = read_site(model, layers, torch.tensor([smp.ids]), Site.DOWN_IN, [1])[1][0].float().cpu()
            a, e = smp.span()
            want = acts[a:e].amax(0) if pool == "max" else acts[e - 1]
            torch.testing.assert_close(st[smp.group].samples[0][0].float(), want, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("device", DEVICES)
def test_paired_sq_diff_self_is_zero(tiny_dir, device):
    tok, model = load_model(tiny_dir, device=device, quantize=False, verbose=False)
    layers = find_layers(model)[0]
    s = [Sample(tok.encode("a b c d e"), 1), Sample(tok.encode("f g"), 0)]
    sq, n = paired_sq_diff(model, layers, model, layers, s, Site.DOWN_IN)
    assert n == 4 + 2 - 0 and sq.shape == (4, W) and sq.abs().max() == 0


# ---- statistics ----------------------------------------------------------------------------------


def test_auroc_and_ap_match_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score

    g = torch.Generator().manual_seed(0)
    pos = torch.randint(0, 5, (40, 6), generator=g).float() + torch.arange(6) * 0.3  # ties on purpose
    neg = torch.randint(0, 5, (55, 6), generator=g).float()
    au = stats.auroc(pos, neg, chunk=4)
    X = torch.cat([pos, neg])
    y = torch.cat([torch.ones(40), torch.zeros(55)])
    for u in range(6):
        assert au[u].item() == pytest.approx(roc_auc_score(y, X[:, u]), abs=1e-6)
    ap = stats.average_precision(X, y.bool(), chunk=4)
    for u in range(6):
        assert ap[u].item() == pytest.approx(average_precision_score(y, X[:, u]), abs=1e-6)
    assert torch.allclose(stats.gini(au), 2 * au - 1)
    # [N, L, W] shapes come back as [L, W]
    assert stats.auroc(pos.reshape(40, 2, 3), neg.reshape(55, 2, 3)).shape == (2, 3)


def test_cohens_d_and_entropy():
    from steering_llms.neurons.collect import GroupStats

    def gs(x):  # x [n, units]
        return GroupStats([0], x.shape[0], x.shape[0], x.sum(0, keepdim=True).double(),
                          (x * x).sum(0, keepdim=True).double(), (x > 0).sum(0, keepdim=True).double())

    a = torch.tensor([[1.0, 0.0], [3.0, 0.0], [2.0, 1.0]])
    b = torch.tensor([[0.0, 0.0], [1.0, 0.0], [-1.0, 1.0]])
    d = stats.cohens_d(gs(a), gs(b))
    want = (a.mean(0) - b.mean(0)) / ((a.var(0, unbiased=False) + b.var(0, unbiased=False)) / 2).sqrt()
    torch.testing.assert_close(d[0], want)
    torch.testing.assert_close(stats.freq_diff(gs(a), gs(b))[0], torch.tensor([2 / 3, 0.0]))
    h = stats.lape_entropy([gs(a), gs(b)])
    assert h[0, 0] < h[0, 1] and h[0, 1] == pytest.approx(math.log(2))
    torch.testing.assert_close(stats.sample_cohens_d(a, b), (a.mean(0) - b.mean(0)) / ((a.var(0) + b.var(0)) / 2).sqrt())


def test_quantiles():
    x = torch.randn(1000, 3)
    torch.testing.assert_close(stats.quantile_cols(x, 0.9), torch.quantile(x, 0.9, dim=0))
    flat = torch.arange(101.0)
    assert stats.flat_quantile(flat, 0.995) == 100.0 and stats.flat_quantile(flat, 0.5) == 50.0


def test_l1_logistic_finds_the_signal():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(400, 50, generator=g)
    y = (X[:, 7] - 0.7 * X[:, 21] + 0.1 * torch.randn(400, generator=g)) > 0
    w, _ = stats.l1_logistic(X, y, lam=0.02, steps=400)
    top = w.abs().topk(2).indices.tolist()
    assert set(top) == {7, 21} and w[7] > 0 > w[21]
    assert (w.abs() > 1e-6).sum() < 20  # sparse


# ---- selection -----------------------------------------------------------------------------------


def test_selection():
    s = torch.tensor([[0.1, -0.9, 0.3, 0.0], [0.5, 0.2, -0.4, 0.8]])
    m = select.top_k_mask(s, 3)
    assert m.sum() == 3 and m[0, 1] and m[1, 3] and m[1, 0]
    assert select.top_k_mask(s, 1, per_layer=True).sum(1).tolist() == [1, 1]
    assert select.threshold_mask(s, 0.45).sum() == 3
    capped = select.cap_per_layer(select.threshold_mask(s, 0.15), s, 1)
    assert capped.sum(1).tolist() == [1, 1] and capped[0, 1] and capped[1, 3]
    assert select.cap_total(torch.ones_like(s, dtype=torch.bool), s, 2).sum() == 2
    ns = select.to_set(m, s, [4, 9], "down_in", {"x": s * 10}, "lab")
    assert ns.keys() == [(4, 1), (9, 3), (9, 0)] and ns.extras["x"] == pytest.approx([-9.0, 8.0, 5.0])
    assert torch.equal(select.to_mask(ns, [4, 9], 4), m)
    qm = select.quantile_mask(torch.arange(200.0).reshape(2, 100), 0.95)
    assert qm.sum(1).tolist() == [5, 5]
    assert select.frequent([m, m, ~m], 0.6).sum() == 3
    sets = [NeuronSet("act", [1, 1], [2, 3], [1, 1]), NeuronSet("act", [1], [2], [1]), NeuronSet("act", [1], [4], [1])]
    assert select.consensus(sets, 0.5) == [(1, 2)]


# ---- attribution ---------------------------------------------------------------------------------


def test_patching_noise_equal_clean_is_zero(f32_model):
    tok, model, layers = f32_model
    ids = ids_of(tok)
    metric = logprob_metric(tok.encode(" cat")[0])
    est, _ = atp(model, layers, ids, ids, metric, layer_ids=[1, 2])
    assert est.abs().max() == 0
    eff, _ = exact_patch(model, layers, ids, ids, metric, [(1, None, 3), (2, 4, 5)])
    assert eff.abs().max() < 1e-5


def test_atp_tracks_exact_patching(f32_model):
    tok, model, layers = f32_model
    clean = ids_of(tok, "The capital of France is the city of")
    noise = ids_of(tok, "The capital of Spain is the city of")
    assert clean.shape == noise.shape
    metric = logprob_metric(tok.encode(" Paris")[0])
    est, _ = atp(model, layers, clean, noise, metric, layer_ids=[0, 1, 2, 3])
    per_neuron = est.sum(1)  # patch every position of a neuron
    top = per_neuron.abs().flatten().topk(40).indices
    cands = [(int(i // W), None, int(i % W)) for i in top]
    eff, _ = exact_patch(model, layers, clean, noise, metric, cands, batch_size=8)
    pred = per_neuron.flatten()[top]
    r = torch.corrcoef(torch.stack([pred, eff]))[0, 1].item()
    assert r > 0.8, r


def test_integrated_gradients_completeness(f32_model):
    tok, model, layers = f32_model
    ids = ids_of(tok, "The capital of France is")
    tgt = tok.encode(" Paris")[0]
    metric = lambda lg: lg.float().softmax(-1)[:, tgt]  # noqa: E731
    ig = integrated_gradients(model, layers, ids, metric, layer_ids=[2], steps=64)
    full = metric(model(input_ids=ids).logits[:, -1]).item()
    with Intervention(layers, [Edit(2, list(range(W)), "set", 0.0, Site.DOWN_IN, scope="last")]):
        zero = metric(model(input_ids=ids).logits[:, -1]).item()
    assert ig.sum().item() == pytest.approx(full - zero, rel=0.1, abs=1e-7)


# ---- decoding and evaluation ---------------------------------------------------------------------


def test_dola_runs(cpu_model):
    tok, model, layers = cpu_model
    out = dola_generate(model, tok, "Say hi.", premature_layers=[0, 1], max_new_tokens=5)
    assert isinstance(out, str)
    with Intervention(layers, [Edit(1, [1, 2], "mul", 0.0, Site.ACT)]):
        assert isinstance(dola_generate(model, tok, "Say hi.", max_new_tokens=3), str)


def test_text_metrics():
    assert refusal_rate(["I'm sorry, but I can't help with that.", "Sure! Here is how."]) == 0.5
    r = lexicon_ratio(["She said her brother was home.", "They left."])
    assert r["female"] == pytest.approx(0.5) and r["male"] == pytest.approx(0.25)
    assert distinct_n(["a b a b a b"], 2) == pytest.approx(2 / 5)
    ns = NeuronSet("act", [1, 1, 2], [3, 4, 5], [1, 2, 3])
    rnd = random_like(ns, W, seed=0)
    assert rnd.layer == ns.layer and not set(rnd.keys()) & set(ns.keys())


def test_batched_generation_matches_single(f32_model):
    from steering_llms.discovery import generate_answer
    from steering_llms.neurons.gen import generate_batch

    tok, model, layers = f32_model
    prompts = ["Hi", "Tell me a long story about a dragon and a knight.", "What is 2+2?"]
    batch = generate_batch(model, tok, prompts, max_new_tokens=8, batch_size=3)
    single = [generate_answer(model, tok, p, max_new_tokens=8, repetition_penalty=None) for p in prompts]
    assert batch == single
    with Intervention(layers, [Edit(2, list(range(40)), "mul", 0.0, Site.ACT, scope="decode")]):
        steered = generate_batch(model, tok, prompts, max_new_tokens=8, batch_size=3)
    assert [s[:1] for s in steered] == [s[:1] for s in single]  # the first token comes from the prefill
