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
    assert json.loads((run / "meta.json").read_text())["env"]["precision"] == "bfloat16"
