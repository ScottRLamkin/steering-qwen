"""``neurons search`` (HedwigAI's pipeline) end to end on the tiny model, with our trait set."""

import json

from steering_llms.neurons import cli
from steering_llms.neurons.traits import NEW, TRAIT_SETS


def test_trait_sets():
    assert len(NEW) == 10 and not set(NEW) & set(TRAIT_SETS["hedwig"])
    assert all(p.endswith("Never break character.") for p in NEW.values())


def test_search_runs_on_tiny_model(tiny_dir, tmp_path):
    run = tmp_path / "search"
    cli.main(["search", "all", "--run-dir", str(run), "--model", tiny_dir, "--device", "cpu", "--no-quantize",
              "--trait-set", "new", "--only-traits", "pirate,terse", "--n-questions", "2", "--n-heldout", "1",
              "--max-new-tokens", "4", "--verify-max-new-tokens", "3", "--answer-tokens", "4", "--top", "2",
              "--screen-questions", "2", "--verify-top", "1", "--layers", "0:4"])
    cfg = json.loads((run / "config.json").read_text())
    assert list(cfg["traits"]) == ["pirate", "terse"]
    answers = json.loads((run / "answers.json").read_text())
    assert set(answers) == {"plain", "pirate", "terse"}
    screen = json.loads((run / "screen.json").read_text())
    assert set(screen["ranked"]) == {"pirate", "terse"} and screen["rows"]
    assert "unsteered" in json.loads((run / "verify.json").read_text())["conditions"]
    ratings = json.loads((run / "ratings.json").read_text())["conditions"]
    assert set(ratings) == set(json.loads((run / "verify.json").read_text())["conditions"])
    assert all(1 <= r["coherence"] <= 5 for r in ratings.values())
    assert set(ratings["unsteered"]["trait"]) == {"pirate", "terse"}
    assert json.loads((run / "meta.json").read_text())["env"]["precision"] == "bfloat16"


def test_hedwig_fluent_only_skips_broken(tiny_dir, tmp_path):
    from steering_llms.neurons.neuronset import load_sets

    run = tmp_path / "search"
    run.mkdir()
    top = {"terse": [{"layer": 2, "neuron": 5, "dose": 1.0, "specificity": 2.9, "rank": 0},
                     {"layer": 1, "neuron": 7, "dose": -1.0, "specificity": 2.7, "rank": 1}]}
    (run / "screen.json").write_text(json.dumps({"top": top}))
    (run / "medians.json").write_text(json.dumps({str(i): 1.5 for i in range(4)}))
    (run / "ratings.json").write_text(json.dumps({"conditions": {
        "L2N5+": {"broken": True, "coherence": 1.4}, "L1N7-": {"broken": False, "coherence": 3.9}}}))
    for flag, first in (([], (2, 5)), (["--fluent-only"], (1, 7))):
        out = tmp_path / f"hedwig{len(flag)}"
        cli.main(["hedwig", "discover", "--model", tiny_dir, "--device", "cpu", "--no-quantize", "--run-dir", str(out),
                  "--hedwig-run", str(run), "--quiet", *flag])
        ns = load_sets(out / "sets.json")["terse"]
        assert (ns.layer[0], ns.index[0]) == first
