"""Every recipe end to end (discover, then evaluate) on the tiny random Qwen3.5 on CPU, with tiny
settings. Checks plumbing, not results: a random 4-layer model has no personality.

Needs the recipes' datasets (a few MB each, cached after the first run); classifiers are faked.
"""

import json

import pytest

from steering_llms.neurons import cli, methods
from steering_llms.neurons.neuronset import load_sets

SMALL = {
    "dpn-le": ["--traits", "Extraversion", "--n-questions", "3"],
    "pcn": ["--traits", "Openness", "--k", "5"],
    "npti": ["--traits", "Extraversion", "--n-questions", "2", "--max-new-tokens", "8", "--min-required", "20"],
    "iris": ["--n-topics", "2", "--n-questions", "2", "--max-new-tokens", "8", "--min-required", "20", "--n-eval-per-topic", "1"],
    "sneuron-tst": ["--n-texts", "10", "--n-eval", "2", "--max-new-tokens", "6"],
    "aura": ["--n-texts", "10", "--n-eval", "2", "--max-new-tokens", "6"],
    "gender": ["--n-per", "10", "--max-new-tokens", "6"],
    "cps": ["--max-new-tokens", "4", "--frac", "0.02"],
    "knowledge": ["--facts-per-relation", "1", "--max-prompts", "3", "--steps", "4", "--min-prob", "0"],
    "atp": ["--n-pairs", "2", "--pool", "20", "--full-pairs", "1"],
    "sentiment": ["--n-texts", "20", "--screen", "64", "--k", "4", "--max-new-tokens", "6"],
    "emotion": ["--n-per", "10", "--n-eval", "2"],
    "safety": ["--n-prompts", "3", "--n-eval", "2", "--max-new-tokens", "5"],
    "hedwig": [],
}


class FakeClassifier:
    def __init__(self, name, device=None):
        self.labels = {"toxicity": ["neutral", "toxic"], "sentiment": ["negative", "positive"]}.get(name, ["a", "b"])

    def __call__(self, texts, batch_size=16):
        return [{self.labels[0]: 0.5, self.labels[1]: 0.5} for _ in texts]

    def mean(self, texts, label):
        return 0.5


def test_every_method_is_listed():
    assert set(SMALL) == set(methods.MODULES)


@pytest.mark.parametrize("name", list(SMALL))
def test_recipe_runs_on_tiny_model(name, tiny_dir, tmp_path, monkeypatch):
    mod = methods.get(name)
    if hasattr(mod, "Classifier"):
        monkeypatch.setattr(mod, "Classifier", FakeClassifier)
    extra = list(SMALL[name])
    if name == "safety":
        extra += ["--base-model", tiny_dir]
    if name == "hedwig":
        run = tmp_path / "hedwig-run"
        run.mkdir()
        top = {"poetic": [{"layer": 2, "neuron": 17, "dose": 1.0, "specificity": 0.5}],
               "evil": [{"layer": 1, "neuron": 3, "dose": -1.0, "specificity": 0.4}]}
        (run / "screen.json").write_text(json.dumps({"top": top}))
        (run / "medians.json").write_text(json.dumps({str(i): 1.5 for i in range(4)}))
        extra += ["--hedwig-run", str(run)]
    run_dir = tmp_path / name
    cli.main([name, "all", "--model", tiny_dir, "--device", "cpu", "--no-quantize", "--run-dir", str(run_dir),
              "--batch-size", "4", "--eval-questions", "1", "--quiet", *extra])
    sets = load_sets(run_dir / "sets.json")
    assert sets, "discover produced no sets"
    for ns in sets.values():
        assert all(0 <= L < 4 for L in ns.layer)
    res = json.loads((run_dir / "eval.json").read_text())
    assert res
    if hasattr(mod, "score"):
        assert json.loads((run_dir / "score.json").read_text())


def test_quant_flags():
    p = cli.build_parser()
    assert p.parse_args(["cps", "--quantize"]).quantize == "nf4"
    assert p.parse_args(["cps", "--quant", "int8"]).quantize == "int8"
    assert p.parse_args(["cps", "--no-quantize"]).quantize is False
    assert p.parse_args(["cps"]).quantize is None


@pytest.mark.parametrize("rediscover", [False, True])
def test_replay_reuses_sets_on_nf4(rediscover, tiny_dir, tmp_path, monkeypatch):
    """Replay a bf16 run on the NF4 model: the saved settings come back, the sets are copied unless
    rediscovering, and the new config records the format."""
    src = tmp_path / "bf16" / "cps"
    cli.main(["cps", "all", "--model", tiny_dir, "--device", "cpu", "--no-quantize", "--run-dir", str(src),
              "--batch-size", "4", "--quiet", "--max-new-tokens", "4", "--frac", "0.02"])
    (src / "facts.json").write_text("[1]")  # any other input the evaluation reads is copied too
    dst = tmp_path / "nf4" / "cps"
    argv = ["replay", str(src), "--run-dir", str(dst), "--quant", "nf4", "--set", "max_new_tokens=3"]
    cli.main(argv + (["--rediscover"] if rediscover else []))
    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["quant"] == "nf4" and cfg["args"]["frac"] == 0.02 and cfg["args"]["max_new_tokens"] == 3
    assert cfg["args"]["replayed_from"] == str(src) and cfg["args"]["rediscovered"] is rediscover
    assert json.loads((dst / "eval.json").read_text())
    assert (dst / "facts.json").exists() is not rediscover
    if not rediscover:
        assert (dst / "sets.json").read_text() == (src / "sets.json").read_text()


def test_quant_report(tmp_path):
    from steering_llms.neurons.quant_report import write_quant_report

    def cps(base, m0):
        c = {"refusal_harmful": base, "refusal_benign": 0.1}
        return {"conditions": {"base": c, "m=0": {**c, "refusal_harmful": m0}, "m=2": {**c, "refusal_benign": 0.3},
                               "random m=0": c}}

    for d, ev in [(tmp_path / "methods" / "cps", cps(0.9, 0.2)), (tmp_path / "quant" / "nf4" / "cps", cps(0.8, 0.5))]:
        d.mkdir(parents=True)
        (d / "eval.json").write_text(json.dumps(ev))
        (d / "config.json").write_text(json.dumps({"quant": "nf4", "args": {}}))
    out = tmp_path / "QUANT.md"
    write_quant_report(tmp_path / "methods", tmp_path / "quant", out)
    text = out.read_text()
    assert "| measure | bf16 | NF4 |" in text
    assert "| CPS: refusal of harmful requests, m=0 | 0.90 → 0.20 (0.90) | 0.80 → 0.50 (0.80) |" in text
