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
