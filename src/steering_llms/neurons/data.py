"""Dataset loaders for the recipes. Files are downloaded once and cached.

Hugging Face datasets come through ``huggingface_hub`` (its cache, works with ``HF_HUB_OFFLINE=1``
once fetched); GitHub files are cached under ``~/.cache/steering_llms``. Each loader returns
plain Python lists/dicts and takes ``n`` / ``seed`` so a recipe can subsample reproducibly.
"""

import csv
import gzip
import json
import os
import random
import urllib.request
from pathlib import Path

CACHE = Path(os.environ.get("STEERING_LLMS_CACHE", Path.home() / ".cache" / "steering_llms"))

BIG5 = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
NPTI_RAW = "https://raw.githubusercontent.com/RUCAIBox/NPTI/main/NPTI/dataset"
PARAREL_RAW = "https://raw.githubusercontent.com/yanaiela/pararel/main/data"


def fetch(url: str, name: str | None = None) -> Path:
    """Download ``url`` into the cache once and return the local path."""
    path = CACHE / (name or url.split("://", 1)[1].replace("/", "_"))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "steering-llms"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        tmp.rename(path)
    return path


def hf_file(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset"))


def _jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _csv(path: Path, delimiter: str = ",") -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _sample(rows: list, n: int | None, seed: int) -> list:
    if n is None or n >= len(rows):
        return list(rows)
    return random.Random(seed).sample(rows, n)


# ---- personality: PersonalityBench (NPTI, DPN-LE, IRiS) -------------------------------------------


def personality_descriptions() -> dict[str, dict[str, list[str]]]:
    """``{trait: {"+": [80 high-pole personas], "-": [80 low-pole personas]}}`` (NPTI's description.json).

    Each line of the file is one JSON object with a persona for every key ``Trait`` (high pole)
    and ``Trait_reversed`` (low pole); NPTI draws one at random per question.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for line in fetch(f"{NPTI_RAW}/description.json").read_text().splitlines():
        if not line.strip():
            continue
        for key, text in json.loads(line).items():
            trait, pole = (key[: -len("_reversed")], "-") if key.endswith("_reversed") else (key, "+")
            out.setdefault(trait, {}).setdefault(pole, []).append(text)
    return out


def personality_questions(trait: str, split: str = "search", n: int | None = None, seed: int = 0) -> list[dict]:
    """PersonalityBench situational questions for ``trait``: ``search`` (with BFI facet, topic and
    sign) or ``test`` (question only)."""
    rows = _jsonl(fetch(f"{NPTI_RAW}/{split}/{trait}.json"))
    return _sample(rows, n, seed)


def ipip_big5() -> dict[str, list[tuple[str, int]]]:
    """A public-domain Big Five inventory: 10 IPIP items per domain, ``(statement, key)`` with key
    +1 for items that score high on the domain and -1 for reverse-keyed ones.

    Stands in for BFI-2 (free for research but not redistributable). These are the 50-item
    IPIP-BFFM markers (Goldberg 1992), public domain.
    """
    return {
        "Extraversion": [
            ("Am the life of the party.", 1), ("Don't talk a lot.", -1), ("Feel comfortable around people.", 1),
            ("Keep in the background.", -1), ("Start conversations.", 1), ("Have little to say.", -1),
            ("Talk to a lot of different people at parties.", 1), ("Don't like to draw attention to myself.", -1),
            ("Don't mind being the center of attention.", 1), ("Am quiet around strangers.", -1),
        ],
        "Agreeableness": [
            ("Feel little concern for others.", -1), ("Am interested in people.", 1), ("Insult people.", -1),
            ("Sympathize with others' feelings.", 1), ("Am not interested in other people's problems.", -1),
            ("Have a soft heart.", 1), ("Am not really interested in others.", -1), ("Take time out for others.", 1),
            ("Feel others' emotions.", 1), ("Make people feel at ease.", 1),
        ],
        "Conscientiousness": [
            ("Am always prepared.", 1), ("Leave my belongings around.", -1), ("Pay attention to details.", 1),
            ("Make a mess of things.", -1), ("Get chores done right away.", 1),
            ("Often forget to put things back in their proper place.", -1), ("Like order.", 1),
            ("Shirk my duties.", -1), ("Follow a schedule.", 1), ("Am exacting in my work.", 1),
        ],
        "Neuroticism": [
            ("Get stressed out easily.", 1), ("Am relaxed most of the time.", -1), ("Worry about things.", 1),
            ("Seldom feel blue.", -1), ("Am easily disturbed.", 1), ("Get upset easily.", 1),
            ("Change my mood a lot.", 1), ("Have frequent mood swings.", 1), ("Get irritated easily.", 1),
            ("Often feel blue.", 1),
        ],
        "Openness": [
            ("Have a rich vocabulary.", 1), ("Have difficulty understanding abstract ideas.", -1),
            ("Have a vivid imagination.", 1), ("Am not interested in abstract ideas.", -1),
            ("Have excellent ideas.", 1), ("Do not have a good imagination.", -1), ("Am quick to understand things.", 1),
            ("Use difficult words.", 1), ("Spend time reflecting on things.", 1), ("Am full of ideas.", 1),
        ],
    }


# ---- style and sentiment ---------------------------------------------------------------------------


def paradetox(n: int | None = None, seed: int = 0) -> list[tuple[str, str]]:
    """ParaDetox (s-nlp/paradetox) ``(toxic, neutral)`` paraphrase pairs."""
    rows = _csv(hf_file("s-nlp/paradetox", "train.tsv"), delimiter="\t")
    pairs = [(r["en_toxic_comment"], r["en_neutral_comment"]) for r in rows if r.get("en_toxic_comment")]
    return _sample(pairs, n, seed)


def sst2(split: str = "train", n: int | None = None, seed: int = 0) -> list[tuple[str, int]]:
    """SST-2 ``(sentence, label)`` with label 1 = positive."""
    rows = _parquet(hf_file("stanfordnlp/sst2", f"data/{split}-00000-of-00001.parquet"))
    return _sample([(r["sentence"].strip(), int(r["label"])) for r in rows], n, seed)


EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def emotion(split: str = "train", n: int | None = None, seed: int = 0) -> list[tuple[str, str]]:
    """dair-ai/emotion ``(text, label name)``."""
    rows = _parquet(hf_file("dair-ai/emotion", f"split/{split}-00000-of-00001.parquet"))
    return _sample([(r["text"], EMOTIONS[int(r["label"])]) for r in rows], n, seed)


def gcgender(split: str = "train", n_per: int | None = None, seed: int = 0) -> dict[str, list[str]]:
    """uzw/GCGender sentences by gender label (``male``, ``female``, ``neutral``)."""
    out: dict[str, list[str]] = {}
    for r in _csv(hf_file("uzw/GCGender", f"{split}.csv")):
        out.setdefault(r["Gender"].strip(), []).append(r["Sentences"].strip())
    return {k: _sample(v, n_per, seed) for k, v in sorted(out.items())}


# ---- safety ----------------------------------------------------------------------------------------


def jbb(kind: str = "harmful") -> list[dict]:
    """JailbreakBench JBB-Behaviors: 100 ``harmful`` or 100 matched ``benign`` behaviors (``Goal`` = prompt)."""
    return _csv(hf_file("JailbreakBench/JBB-Behaviors", f"data/{kind}-behaviors.csv"))


def hh_harmless(split: str = "test", n: int | None = None, seed: int = 0) -> list[str]:
    """The first human turn of Anthropic HH-RLHF harmless-base conversations."""
    rows = _jsonl(hf_file("Anthropic/hh-rlhf", f"harmless-base/{split}.jsonl.gz"))
    prompts = []
    for r in rows:
        text = r["chosen"]
        if "\n\nHuman:" in text:
            first = text.split("\n\nHuman:", 1)[1].split("\n\nAssistant:", 1)[0].strip()
            if first:
                prompts.append(first)
    return _sample(list(dict.fromkeys(prompts)), n, seed)


# ---- facts: ParaRel (Knowledge Neurons) -------------------------------------------------------------

PARAREL_RELATIONS = ["P36", "P1376", "P37", "P103", "P364", "P140", "P19", "P20", "P27", "P495", "P17", "P30"]


def pararel(
    relations: list[str] | None = None,
    facts_per_relation: int = 3,
    min_prompts: int = 3,
    seed: int = 0,
) -> list[dict]:
    """ParaRel facts with several paraphrased prompts each, for a left-to-right model.

    Only patterns that end in ``[Y]`` are kept, so the prompt is the text before the answer.
    Returns ``[{"relation", "subject", "answer", "prompts": [...]}]``.
    """
    rng = random.Random(seed)
    out = []
    for rel in relations or PARAREL_RELATIONS:
        try:
            patterns = _jsonl(fetch(f"{PARAREL_RAW}/pattern_data/graphs_json/{rel}.jsonl"))
            facts = _jsonl(fetch(f"{PARAREL_RAW}/trex_lms_vocab/{rel}.jsonl"))
        except Exception:
            continue
        ends = []
        for p in patterns:
            pat = p["pattern"].strip().rstrip(".").strip()
            if pat.endswith("[Y]") and pat.count("[X]") == 1:
                ends.append(pat[: -len("[Y]")].rstrip())
        ends = list(dict.fromkeys(ends))
        if len(ends) < min_prompts:
            continue
        for f in rng.sample(facts, min(facts_per_relation, len(facts))):
            out.append(
                {
                    "relation": rel,
                    "subject": f["sub_label"],
                    "answer": f["obj_label"],
                    "prompts": [e.replace("[X]", f["sub_label"]) for e in ends],
                }
            )
    return out
