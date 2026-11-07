"""The quantization-robustness study: fake quantization, per-question scoring, and every stage end to
end on the tiny model (bf16, fake int3 and NF4)."""

import json

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_llms import Steerer
from steering_llms import discovery as D
from steering_llms.neurons import cli
from steering_llms.neurons.robustness import ItemScorer, fake_quantize
from steering_llms.neurons.robustness_report import partial_spearman, ranks, spearman


def test_fake_quantize_levels_and_idempotent():
    torch.manual_seed(0)
    lin = nn.Linear(128, 8, bias=False)
    w0 = lin.weight.detach().clone()
    assert fake_quantize(lin, 3) == 1
    w = lin.weight.detach().reshape(8, 2, 64)
    scale = w0.reshape(8, 2, 64).abs().amax(-1, keepdim=True) / 3
    q = w / scale
    assert torch.allclose(q, q.round(), atol=1e-4) and q.abs().max() <= 3 + 1e-4
    assert all(len(torch.unique(q[i, j].round())) <= 7 for i in range(8) for j in range(2))
    w1 = lin.weight.detach().clone()
    fake_quantize(lin, 3)
    assert torch.allclose(lin.weight, w1, atol=1e-6)


def test_fake_quantize_8_bit_barely_changes():
    torch.manual_seed(0)
    lin = nn.Linear(256, 32, bias=False)
    w0 = lin.weight.detach().clone()
    fake_quantize(lin, 8)
    rel = (lin.weight - w0).norm() / w0.norm()
    assert 0 < rel < 0.01
    lin2 = nn.Linear(256, 32, bias=False)
    lin2.weight.data.copy_(w0)
    fake_quantize(lin2, 3)
    assert (lin2.weight - w0).norm() / w0.norm() > 5 * rel


def test_item_scorer_matches_set_means(tiny_dir):
    tok = AutoTokenizer.from_pretrained(tiny_dir)
    model = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32).eval()
    qs = ["What is up?", "Tell me a story about a very long river and a boat."]
    answers = {"plain": ["Not much.", "Once there was a river."], "pirate": ["Arr, nothing!", "Arr matey."]}
    a = D.AnswerScorer(model, tok, qs, answers, n_questions=2, answer_tokens=6)
    b = ItemScorer(model, tok, qs, answers, n_questions=2, answer_tokens=6)
    means, items = a(), b.items()
    for k in answers:
        assert len(items[k]) == 2
        assert sum(items[k]) / 2 == pytest.approx(means[k], abs=1e-5)


def test_rank_statistics():
    assert ranks([3.0, 1.0, 2.0, 2.0]) == [3.0, 0.0, 1.5, 1.5]
    assert spearman([1, 2, 3, 4], [10, 20, 30, 45]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # x and y related only through z: the partial correlation is ~0
    g = torch.Generator().manual_seed(0)
    z = torch.randn(200, generator=g)
    x = (z + 0.5 * torch.randn(200, generator=g)).tolist()
    y = (z + 0.5 * torch.randn(200, generator=g)).tolist()
    assert spearman(x, y) > 0.6 and abs(partial_spearman(x, y, z.tolist())) < 0.15


def test_study_end_to_end_on_tiny_model(tiny_dir, tmp_path):
    search = tmp_path / "search"
    cli.main(["search", "all", "--run-dir", str(search), "--model", tiny_dir, "--device", "cpu", "--no-quantize",
              "--trait-set", "new", "--only-traits", "pirate,terse,shy", "--n-questions", "3", "--n-heldout", "1",
              "--max-new-tokens", "5", "--verify-max-new-tokens", "3", "--answer-tokens", "4", "--top", "3",
              "--screen-questions", "3", "--verify-top", "1", "--layers", "0:4"])
    run = tmp_path / "robust"
    common = ["--run-dir", str(run), "--model", tiny_dir, "--device", "cpu"]
    settings = ["--studies", "new", "--study-runs", json.dumps({"new": str(search)}), "--top", "1", "--weak", "2",
                "--n-random", "2", "--n-questions", "3", "--heldout-tokens", "5", "--n-prompts", "2",
                "--reply-tokens", "3", "--layers", "0:4"]
    cli.main(["robustness", "panel,heldout,score,dose,replies,attribute", "--format", "bf16", *common, *settings])
    panel = json.loads((run / "panel.json").read_text())
    assert {e["kind"] for e in panel} == {"top", "weak", "random"}
    st = Steerer(AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.bfloat16), AutoTokenizer.from_pretrained(tiny_dir))
    for e in panel:  # every push, random or not, moves the residual by dose * median norm
        push = abs(e["value"]) * st.down_col(e["layer"], e["neuron"]).norm().item()
        assert push == pytest.approx(e["median_norm"], rel=2e-2)
    for fmt, stages in [("fq3", "score,dose,replies,attribute"), ("nf4", "score,replies,attribute")]:
        cli.main(["robustness", stages, "--format", fmt, *common])
    cli.main(["robustness", "judge,report", "--format", "bf16", *common])
    for fmt in ("bf16", "fq3", "nf4"):
        sc = json.loads((run / fmt / "score_new.json").read_text())
        assert len(sc["shift"]) >= len({e["key"] for e in panel})
        assert (run / fmt / "cols.pt").exists()
        assert "unsteered" in json.loads((run / fmt / "judge.json").read_text())
        assert "watch" in json.loads((run / fmt / "attrib_new.json").read_text())
    assert any(k.endswith("@2") for k in json.loads((run / "fq3" / "score_new.json").read_text())["shift"])
    text = (run / "REPORT.md").read_text()
    assert "## 1. How much each format keeps" in text and "| fake int3 |" in text and "## 7. Every neuron" in text
    assert json.loads((run / "summary.json").read_text())
