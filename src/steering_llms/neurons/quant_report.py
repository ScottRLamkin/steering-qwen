"""Compare the bf16 runs in ``runs/methods`` with their replays on quantized models in ``runs/quant``.

``runs/quant/<fmt>/<recipe>`` evaluates the bf16 run's own neurons on a ``<fmt>`` model ("transfer");
``runs/quant/<fmt>-rediscovered/<recipe>`` finds the neurons again on that model. Each cell is the
headline measure of one recipe in one format, written ``no edit -> edited (random neurons)``.
"""

import json
import statistics
from collections.abc import Callable
from pathlib import Path

from .neuronset import load_sets
from .report import N_NEURONS, _table

FORMATS = ("nf4", "int8", "fp4")
LABEL = {"bf16": "bf16", "nf4": "NF4", "int8": "int8", "fp4": "FP4"}


def _json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def _arrow(base: float, edited: float, rnd: float | None = None, nd: int = 2, pct: bool = False) -> str:
    f = (lambda x: f"{x:.0%}") if pct else (lambda x: f"{x:.{nd}f}")
    return f"{f(base)} → {f(edited)}" + (f" ({f(rnd)})" if rnd is not None else "")


def _shift(x: float, rnd: float | None = None) -> str:
    return f"{x:+.2f}" + (f" ({rnd:+.2f})" if rnd is not None else "")


Cell = Callable[[Path], str | None]


def personality_rows(name: str) -> list[tuple[str, Cell]]:
    def judge(pole: str) -> Cell:
        def f(d: Path):
            ev = _json(d / "eval.json")
            if not ev:
                return None
            return _arrow(_mean(t["base"]["judge"] for t in ev.values()), _mean(t[pole]["judge"] for t in ev.values()),
                          _mean(t["random" + pole]["judge"] for t in ev.values()))
        return f

    def logprob(pole: str) -> Cell:
        def f(d: Path):
            sc = _json(d / "score.json")
            return _shift(_mean(t[pole] for t in sc.values()), _mean(t["random" + pole] for t in sc.values())) if sc else None
        return f

    def ppl(d: Path):
        ev = _json(d / "eval.json")
        if not ev:
            return None
        return _arrow(_mean(t["base"]["perplexity"] for t in ev.values()),
                      _mean((t["+"]["perplexity"] + t["-"]["perplexity"]) / 2 for t in ev.values()), nd=1)

    return [
        (f"{name}: judge 1-5, up push", judge("+")),
        (f"{name}: judge 1-5, down push", judge("-")),
        (f"{name}: log-prob shift, up push", logprob("+")),
        (f"{name}: log-prob shift, down push", logprob("-")),
        (f"{name}: reply perplexity (both pushes)", ppl),
    ]


HEDWIG_NAMED = (("evil", "Agreeableness", -1), ("arrogant", "Openness", -1), ("humorous", "Extraversion", 1),
                ("cheerful", "Extraversion", 1))


def hedwig(d: Path):
    sc = _json(d / "score.json")
    if not sc:
        return None
    named = _mean(s * sc[t][n] for n, t, s in HEDWIG_NAMED)
    rnd = _mean(abs(sc[t].get(f"{n} random", 0.0)) for n, t, _ in HEDWIG_NAMED)
    return f"{named:+.2f} (±{rnd:.2f})"


def cond(file_key: str, base: str, edited: str, rnd: str | None = None, pct: bool = False, nd: int = 2) -> Cell:
    def f(d: Path):
        ev = _json(d / "eval.json")
        if not ev:
            return None
        c = ev["conditions"]
        return _arrow(c[base][file_key], c[edited][file_key], c[rnd][file_key] if rnd else None, nd=nd, pct=pct)
    return f


def knowledge(key: str, median: bool = False) -> Cell:
    def f(d: Path):
        ev = _json(d / "eval.json")
        if not ev:
            return None
        if median:
            return f"{statistics.median(r[key] for r in ev['facts']):+.0%}"
        if key == "suppress":
            return f"{ev['mean']['suppress']:+.0%} ({ev['mean']['random_suppress']:+.0%})"
        return f"{ev['mean'][key]:+.0%}"
    return f


def safety_gap(d: Path):
    ev = _json(d / "eval.json")
    if not ev:
        return None
    c = {k: v["harm"] for k, v in ev["conditions"].items()}
    gap = c["base"] - c["instruct"]
    if abs(gap) < 1e-6:
        return None
    return f"{(c['base'] - c['base+patch']) / gap:.0%} ({(c['base'] - c['base+random patch']) / gap:.0%}) of a {gap:.2f} gap"


ROWS: list[tuple[str, str, Cell]] = [
    *[("npti", label, f) for label, f in personality_rows("NPTI")],
    *[("dpn-le", label, f) for label, f in personality_rows("DPN-LE")],
    ("hedwig", "HedwigAI: log-prob shift along each neuron's named trait (evil, arrogant, humorous, cheerful)", hedwig),
    ("cps", "CPS: refusal of harmful requests, m=0", cond("refusal_harmful", "base", "m=0", "random m=0")),
    ("cps", "CPS: refusal of benign requests, m=2", cond("refusal_benign", "base", "m=2")),
    ("knowledge", "Knowledge: answer probability, neurons zeroed", knowledge("suppress")),
    ("knowledge", "Knowledge: answer probability, neurons doubled (median)", knowledge("amplify", median=True)),
    ("knowledge", "Knowledge: other relations' facts, neurons zeroed", knowledge("other_facts_suppress")),
    ("aura", "AurA: toxicity of RTP continuations", cond("toxicity", "base", "aura", "random")),
    ("aura", "AurA: perplexity of neutral text", cond("neutral_ppl", "base", "aura", "random", nd=0)),
    ("sentiment", "Sentiment: P(positive), top unit clamped high x4", cond("positive", "base", "+top1 x4")),
    ("sentiment", "Sentiment: P(positive), top unit clamped low x4", cond("positive", "base", "-top1 x4")),
    ("sentiment", "Sentiment: P(positive), top 16 clamped high", cond("positive", "base", "+top16", "random+top16")),
    ("safety", "Safety: harm of Base replies, instruct values patched in", cond("harm", "base", "base+patch", "base+random patch")),
    ("safety", "Safety: share of the Base-to-Instruct harm gap closed", safety_gap),
]


def _columns(base: Path, quant: Path) -> list[tuple[str, Callable[[str], Path]]]:
    cols: list[tuple[str, Callable[[str], Path]]] = [("bf16", lambda r: base / r)]
    for fmt in FORMATS:
        if (quant / fmt).is_dir():
            cols.append((LABEL[fmt], lambda r, fmt=fmt: quant / fmt / r))
    for fmt in FORMATS:
        if (quant / f"{fmt}-rediscovered").is_dir() and any((quant / f"{fmt}-rediscovered" / r).is_dir() for r, _, _ in ROWS):
            cols.append((f"{LABEL[fmt]}, found again", lambda r, fmt=fmt: quant / f"{fmt}-rediscovered" / r))
    return cols


def overlap_rows(base: Path, quant: Path) -> list[list]:
    rows = []
    for fmt in FORMATS:
        root = quant / f"{fmt}-rediscovered"
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if (p / "sets.json").exists() and p.name != "atp"):
            src = base / d.name / "sets.json"
            if not src.exists():
                continue
            a, b = load_sets(src), load_sets(d / "sets.json")
            keys = [k for k in a if k in b and len(a[k]) and len(b[k])]
            if not keys:
                continue
            jac = _mean(a[k].jaccard(b[k]) for k in keys)
            n = _mean((len(a[k]) + len(b[k])) / 2 for k in keys)
            total = N_NEURONS if a[keys[0]].site.value != "down_out" else 32 * 2560
            p = n / total
            rows.append([LABEL[fmt], d.name, len(keys), f"{_mean(len(a[k]) for k in keys):,.0f}",
                         f"{_mean(len(b[k]) for k in keys):,.0f}", f"{jac:.2f}", f"{p / (2 - p):.3f}"])
    return rows


def atp_rows(quant: Path, bf16_top: dict[str, set]) -> list[list]:
    rows = []
    for fmt in ("bf16", *FORMATS):
        ev = _json(quant / f"{fmt}-rediscovered" / "atp" / "eval.json")
        if not ev:
            continue
        rs = list(ev.values())
        same = [len({(t["layer"], t["neuron"]) for t in r["top_true"]} & bf16_top[k]) for k, r in ev.items() if k in bf16_top]
        rows.append([LABEL[fmt], len(rs), f"{_mean(r['pearson'] for r in rs):.3f}", f"{_mean(r['recall@10'] for r in rs):.2f}",
                     f"{_mean(r['recall@30'] for r in rs):.2f}", f"{_mean(r['recall@100'] for r in rs):.2f}",
                     f"{_mean(r['sign_agree_top100'] for r in rs):.2f}", f"{_mean(same):.1f} / 10" if same else "–"])
    return rows


def write_quant_report(base: Path, quant: Path, out_path: Path) -> None:
    base, quant = Path(base), Path(quant)
    cols = _columns(base, quant)
    lines = ["# Neuron methods on quantized Qwen3.5-4B", ""]
    notes = quant / "NOTES.md"
    if notes.exists():
        lines += [notes.read_text().strip(), ""]
    lines += [
        "## Same neurons, other formats",
        "",
        "Each recipe's bf16 run, replayed with the same settings. In the format columns the neurons and edits are "
        "the bf16 run's own, so a smaller effect means the quantized model responds less to the same edit. "
        "The *found again* columns rerun discovery on the quantized model first. Cells read "
        "`no edit → edited (random neurons, same edit)`; log-prob shifts are already relative to no edit. "
        "Every format measures its own no-edit baseline, and the personality judge and the Safety harm rating "
        "are that format's own model, so compare each cell with its baseline and random control first.",
        "",
    ]
    body = []
    for recipe, label, fn in ROWS:
        cells = [fn(path(recipe)) if path(recipe).is_dir() else None for _, path in cols]
        if any(c is not None for c in cells):
            body.append([label, *[c if c is not None else "–" for c in cells]])
    lines += _table(["measure", *[h for h, _ in cols]], body) + [""]

    base_atp = _json(quant / "bf16-rediscovered" / "atp" / "eval.json") or {}
    top = {k: {(t["layer"], t["neuron"]) for t in r["top_true"]} for k, r in base_atp.items()}
    arows = atp_rows(quant, top)
    if arows:
        lines += ["## AtP against exact patching", "",
                  "AtP found again in each format and checked by exact patching on pools (AtP's top 300 plus 300 random "
                  "neurons) for 4 country pairs; means over the pairs. The last column counts how many of each pair's 10 "
                  "neurons with the largest exact effect are also among bf16's 10.", ""]
        lines += _table(["format", "pairs", "pearson", "recall@10", "recall@30", "recall@100", "sign agreement",
                         "true top 10 shared with bf16"], arows) + [""]

    orows = overlap_rows(base, quant)
    if orows:
        lines += ["## Do the quantized models have the same neurons?", "",
                  "Jaccard overlap between the neuron sets found in bf16 and those found again on the quantized model, "
                  "averaged over each recipe's sets (traits, poles, facts or sites). **chance** is the overlap of two "
                  "random sets of the same size.", ""]
        lines += _table(["format", "recipe", "sets", "neurons per set (bf16)", "neurons per set (found again)",
                         "Jaccard", "chance"], orows) + [""]

    status = []
    for root in sorted(p for p in quant.iterdir() if p.is_dir()):
        for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "_shared"):
            cfg = _json(d / "config.json") or {}
            status.append([f"`{root.name}/{d.name}`", "ran" if (d / "eval.json").exists() else "not finished",
                           cfg.get("quant") or "bf16"])
    if status:
        lines += ["## Runs", "", *_table(["run", "status", "format"], status), ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
