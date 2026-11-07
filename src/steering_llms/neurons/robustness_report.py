"""``runs/robustness/REPORT.md``: how much of each neuron's steering survives each format, and
whether that tracks how strong and clean the neuron's meaning is in bf16.

Definitions (per study, per format, per panel neuron of trait ``t``):

* ``shift[s][q]``: change in mean log-prob of answer set ``s``'s answer to held-out question ``q``.
* specificity ``spec[q] = shift[t][q] - mean over the study's other traits o of shift[o][q]``
  (HedwigAI's screen score, per question). ``eff`` is its mean over questions.
* the random neurons' ``eff`` for every trait of the study is the null distribution; ``z`` is a
  neuron's ``eff`` against it.
* retention ``R_f = eff_f / eff_bf16``, for neurons whose bf16 interval excludes 0; ``d_f = eff_f - eff_bf16``.
  Intervals are a bootstrap over questions, resampled the same way in every format (paired).

bf16 measures of meaning, none of which uses a quantized model:

* ``eff`` and ``z``
* consistency: share of questions with ``spec[q] > 0``
* selectivity: ``shift[t]`` minus the largest ``shift[o]`` (another trait's set)
* judge gain: the bf16 judge's rating of the trait in the neuron's replies, minus in unsteered replies
* coherence: the bf16 judge's 1-5 rating of the neuron's replies as fluent, coherent English, whatever
  their style. A neuron is *broken* when its mean is under ``BROKEN_COH`` (gibberish, loops, switching
  language), else *fluent*. On the searches' held-out replies this threshold separates every reply read
  as broken by hand (at most 2.44) from the fluent ones (mostly 2.8 and up).
* fluency cost: log of (perplexity of the neuron's replies / of the unsteered replies), both under the
  unsteered bf16 model. Loops have low perplexity, so this is a weaker guide than coherence.
* reproduced (HedwigAI traits): in the top 10 for the trait in the MLX 8-bit and torch NF4 searches
"""

import json
import math
import random
from pathlib import Path

import torch

from .. import discovery as D
from .evaluate import distinct_n
from .traits import KIND

FORMAT_ORDER = ["bf16", "nf4", "int8", "fp4", "fq4", "fq3"]
LABEL = {"bf16": "bf16", "nf4": "NF4", "int8": "int8", "fp4": "FP4", "fq4": "fake int4", "fq3": "fake int3"}
B = 2000
BROKEN_COH = 2.5


def _j(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


# ---- statistics ------------------------------------------------------------------------------------


def ranks(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy) if sx and sy else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


def residual(y: list[float], z: list[float]) -> list[float]:
    n = len(y)
    mz, my = sum(z) / n, sum(y) / n
    vz = sum((a - mz) ** 2 for a in z)
    b = sum((a - mz) * (c - my) for a, c in zip(z, y)) / vz if vz else 0.0
    return [c - my - b * (a - mz) for a, c in zip(z, y)]


def partial_spearman(x: list[float], y: list[float], z: list[float]) -> float:
    """Spearman correlation of x and y with z's (rank) linear effect removed from both."""
    rz = ranks(z)
    return pearson(residual(ranks(x), rz), residual(ranks(y), rz))


def perm_p(x: list[float], y: list[float], stat=spearman, n: int = 1000, seed: int = 0) -> float:
    """Two-sided permutation p-value of ``stat(x, y)``."""
    obs = abs(stat(x, y))
    if math.isnan(obs):
        return float("nan")
    rng = random.Random(seed)
    y2 = list(y)
    hit = 0
    for _ in range(n):
        rng.shuffle(y2)
        if abs(stat(x, y2)) >= obs - 1e-12:
            hit += 1
    return (hit + 1) / (n + 1)


def mean(x) -> float:
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(x) / len(x) if x else float("nan")


def median(x) -> float:
    x = sorted(v for v in x if v is not None and not math.isnan(v))
    if not x:
        return float("nan")
    m = len(x) // 2
    return x[m] if len(x) % 2 else (x[m - 1] + x[m]) / 2


def sd(x) -> float:
    x = [v for v in x if v is not None and not math.isnan(v)]
    if len(x) < 2:
        return float("nan")
    m = sum(x) / len(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


# ---- loading ---------------------------------------------------------------------------------------


def spec_q(shift: dict[str, list[float]], trait: str, traits: list[str]) -> list[float]:
    others = [o for o in traits if o != trait]
    return [shift[trait][q] - sum(shift[o][q] for o in others) / len(others) for q in range(len(shift[trait]))]


def other_run_ranks(path: Path) -> dict | None:
    """``{trait: {(L, n, dose): rank}}`` from another search run's screen (torch or MLX format)."""
    screen = _j(path / "screen.json") or _j(path / "neurons.json.screen.json")
    if not screen:
        return None
    rows = screen["rows"]
    traits = [k for k in rows[0]["shift"] if k != D.PLAIN]
    ranked = D.rank_candidates(rows, traits, top=0)
    return {t: {(x["layer"], x["neuron"], float(x["dose"])): i for i, x in enumerate(v)} for t, v in ranked.items()}


class Data:
    def __init__(self, run_dir: Path) -> None:
        self.dir = Path(run_dir)
        self.cfg = _j(self.dir / "config.json")
        self.panel = _j(self.dir / "panel.json")
        self.studies = self.cfg["studies"]
        found = [p.name for p in self.dir.iterdir() if p.is_dir() and any(p.glob("score_*.json"))]
        self.formats = [f for f in FORMAT_ORDER if f in found] + sorted(f for f in found if f not in FORMAT_ORDER)
        self.score = {f: {s: _j(self.dir / f / f"score_{s}.json") for s in self.studies} for f in self.formats}
        self.traits = {}
        for s in self.studies:
            h = _j(self.dir / "_shared" / f"heldout_{s}.json")
            self.traits[s] = [k for k in h["answers"] if k != D.PLAIN]
        self.Q = len(_j(self.dir / "_shared" / f"heldout_{self.studies[0]}.json")["questions"])
        g = torch.Generator().manual_seed(0)
        self.boot = torch.randint(0, self.Q, (B, self.Q), generator=g)
        self.judge = {f: _j(self.dir / f / "judge.json") for f in self.formats}
        self.replies = {f: _j(self.dir / f / "replies.json") for f in self.formats}
        self.fluency = {f: _j(self.dir / f / "fluency.json") for f in self.formats}
        self.coherence = {f: _j(self.dir / f / "coherence.json") for f in self.formats}
        self.attrib = {f: {s: _j(self.dir / f / f"attrib_{s}.json") for s in self.studies} for f in self.formats}
        self.cols = {}
        for f in self.formats:
            p = self.dir / f / "cols.pt"
            if p.exists():
                self.cols[f] = torch.load(p)

    def shift(self, f: str, study: str, k: str) -> dict | None:
        sc = self.score[f][study]
        return sc["shift"].get(k) if sc else None

    def spec(self, f: str, e: dict, trait: str | None = None, mult: float = 1.0) -> list[float] | None:
        from .robustness import key

        sh = self.shift(f, e["study"], key(e["layer"], e["neuron"], e["dose"], mult))
        return spec_q(sh, trait or e["trait"], self.traits[e["study"]]) if sh else None

    def boot_means(self, q: list[float]) -> torch.Tensor:
        return torch.tensor(q)[self.boot].mean(1)

    def null(self, f: str, study: str) -> list[float]:
        out = []
        for e in self.panel:
            if e["study"] == study and e["kind"] == "random":
                for t in self.traits[study]:
                    s = self.spec(f, e, t)
                    if s:
                        out.append(mean(s))
        return out

    def judge_gain(self, f: str, e: dict) -> float | None:
        j = self.judge.get(f)
        if not j or e["key"] not in j or e["trait"] not in j[e["key"]] or e["trait"] not in j.get("unsteered", {}):
            return None
        return mean(j[e["key"]][e["trait"]]) - mean(j["unsteered"][e["trait"]])

    def fluency_cost(self, f: str, k: str) -> float | None:
        fl = self.fluency.get(f)
        if not fl or k not in fl or "unsteered" not in fl:
            return None
        return math.log(fl[k] / fl["unsteered"])

    def coh(self, f: str, k: str) -> float | None:
        c = self.coherence.get(f)
        return mean(c[k]) if c and k in c else None

    def distinct2(self, f: str, k: str) -> float | None:
        r = self.replies.get(f)
        return distinct_n(r[k], 2) if r and k in r else None

    def attrib_rank(self, f: str, e: dict) -> tuple[int, bool] | None:
        """Best rank over the trait's two shortlists (0 = top of 221k) and whether that score's sign
        matches the neuron's dose."""
        a = self.attrib[f][e["study"]]
        if not a or "watch" not in a:
            return None
        best = None
        for ref in ("vs_plain", "vs_others"):
            w = a["watch"].get(f"{e['trait']}:{ref}", {}).get(f"L{e['layer']}N{e['neuron']}")
            if w and (best is None or w["rank"] < best[0]):
                best = (w["rank"], (w["score"] > 0) == (e["dose"] > 0))
        return best

    def col_err(self, f: str, e: dict) -> tuple[float, float] | None:
        if f not in self.cols or "bf16" not in self.cols:
            return None
        k = f"L{e['layer']}N{e['neuron']}"
        a, b = self.cols["bf16"].get(k), self.cols[f].get(k)
        if a is None or b is None:
            return None
        cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        return 1 - cos, (b.norm() / a.norm()).item()


# ---- the analysis ----------------------------------------------------------------------------------


def neuron_records(d: Data) -> list[dict]:
    repro_runs = {"mlx-8bit": Path("runs/mlx-8bit"), "torch-nf4": Path("runs/torch-nf4")}
    other = {k: other_run_ranks(p) for k, p in repro_runs.items() if p.exists()}
    recs = []
    nulls = {(f, s): d.null(f, s) for f in d.formats for s in d.studies}
    for e in d.panel:
        if e["kind"] == "random":
            continue
        r = {k: e[k] for k in ("study", "trait", "kind", "key", "layer", "neuron", "dose", "value", "screen_rank", "screen_spec")}
        r["trait_kind"] = KIND.get(e["trait"], "other")
        base = d.spec("bf16", e)
        if base is None:
            continue
        bb = d.boot_means(base)
        sh = d.shift("bf16", e["study"], e["key"])
        others = [o for o in d.traits[e["study"]] if o != e["trait"]]
        nm = nulls[("bf16", e["study"])]
        r.update(
            eff=mean(base),
            lo=bb.quantile(0.025).item(),
            hi=bb.quantile(0.975).item(),
            z=(mean(base) - mean(nm)) / sd(nm) if len(nm) > 1 else float("nan"),
            consistency=mean([1.0 if v > 0 else 0.0 for v in base]),
            selectivity=mean(sh[e["trait"]]) - max(mean(sh[o]) for o in others),
            judge_gain=d.judge_gain("bf16", e),
            fluency_cost=d.fluency_cost("bf16", e["key"]),
            distinct2=d.distinct2("bf16", e["key"]),
            coherence=d.coh("bf16", e["key"]),
        )
        r["significant"] = r["lo"] > 0
        if r["coherence"] is not None:
            r["broken"] = r["coherence"] < BROKEN_COH
        if e["study"] == "hedwig":
            hits = []
            for k, rk in other.items():
                if rk and e["trait"] in rk:
                    i = rk[e["trait"]].get((e["layer"], e["neuron"], float(e["dose"])))
                    hits.append(1.0 if i is not None and i < 10 else 0.0)
            r["reproduced"] = sum(hits) if hits else None
        r["formats"] = {}
        for f in d.formats:
            q = d.spec(f, e)
            if q is None:
                continue
            fb = d.boot_means(q)
            diff = fb - bb
            nf = nulls[(f, e["study"])]
            fr = {"eff": mean(q), "d": mean(q) - mean(base), "d_se": diff.std().item(),
                  "z": (mean(q) - mean(nf)) / sd(nf) if len(nf) > 1 else float("nan")}
            if r["significant"]:
                fr["R"] = mean(q) / mean(base)
                ratio = fb / bb
                fr["R_lo"], fr["R_hi"] = ratio.quantile(0.025).item(), ratio.quantile(0.975).item()
            fr["judge_gain"] = d.judge_gain(f, e)
            fr["fluency_cost"] = d.fluency_cost(f, e["key"])
            fr["coherence"] = d.coh(f, e["key"])
            ar = d.attrib_rank(f, e)
            if ar:
                fr["attrib_rank"], fr["attrib_sign"] = ar
            ce = d.col_err(f, e)
            if ce:
                fr["col_err"], fr["col_norm_ratio"] = ce
            r["formats"][f] = fr
        recs.append(r)
    return recs


def coh_cell(r: dict) -> str:
    c = r.get("coherence")
    if c is None:
        return "–"
    return f"{c:.1f}" + (" ✗" if r.get("broken") else "")


def fmt(x, nd=2, sign=True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def write_report(run_dir: Path) -> None:
    d = Data(run_dir)
    recs = neuron_records(d)
    quant = [f for f in d.formats if f != "bf16"]
    sig = [r for r in recs if r["significant"]]
    L: list[str] = []
    w = L.append
    w("# Does surviving quantization measure a neuron's meaning?\n")
    w(f"Panel: {len(recs)} neurons over {sum(len(d.traits[s]) for s in d.studies)} traits "
      f"({', '.join(f'{s}: {len(d.traits[s])}' for s in d.studies)}), plus "
      f"{sum(1 for e in d.panel if e['kind'] == 'random')} random neurons. "
      f"{len(sig)} neurons have bf16 specificity above 0 (bootstrap 95% interval over {d.Q} held-out questions). "
      f"Formats: {', '.join(LABEL.get(f, f) for f in d.formats)}.\n")
    w("Definitions are in `src/steering_llms/neurons/robustness_report.py`. Retention R = specificity in the format / "
      "specificity in bf16, for the significant neurons; the edit is the same `down_proj`-input value in every format.\n")

    # 1. formats overview
    w("## 1. How much each format keeps\n")
    w("| format | neurons | mean spec | random spread (sd) | median R | R 10–90% | slope vs bf16 | r vs bf16 | "
      "still beat random (z>2) | spread beyond noise (sd of d) | plain-answer log-prob |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in d.formats:
        effs = [r["formats"][f]["eff"] for r in recs if f in r["formats"]]
        base_effs = [r["eff"] for r in recs if f in r["formats"]]
        Rs = [r["formats"][f]["R"] for r in sig if f in r["formats"]]
        nulls = [v for s in d.studies for v in d.null(f, s)]
        zs = [r["formats"][f]["z"] for r in sig if f in r["formats"]]
        ds = [r["formats"][f]["d"] for r in sig if f in r["formats"]]
        ses = [r["formats"][f]["d_se"] for r in sig if f in r["formats"]]
        excess = math.sqrt(max(0.0, sd(ds) ** 2 - mean([s * s for s in ses]))) if len(ds) > 2 and f != "bf16" else float("nan")
        slope = sum(a * b for a, b in zip(base_effs, effs)) / sum(b * b for b in base_effs) if base_effs else float("nan")
        plain = mean([mean(d.score[f][s]["base"][D.PLAIN]) for s in d.studies if d.score[f][s]])
        q = sorted(Rs)
        w(f"| {LABEL.get(f, f)} | {len(effs)} | {fmt(mean(effs))} | {fmt(sd(nulls), sign=False)} | {fmt(median(Rs))} | "
          f"{fmt(q[int(0.1 * (len(q) - 1))]) if q else '–'} – {fmt(q[int(0.9 * (len(q) - 1))]) if q else '–'} | "
          f"{fmt(slope)} | {fmt(pearson(base_effs, effs))} | {sum(z > 2 for z in zs)}/{len(zs)} | {fmt(excess, sign=False)} | {fmt(plain)} |")
    w("\n*spread beyond noise*: the sd of `d = eff_f − eff_bf16` across neurons left after subtracting the "
      "bootstrap noise of each `d`; 0 means every neuron loses the same amount, within noise.\n")

    # 2. fake vs real
    if "fq4" in d.formats:
        w("## 2. Is fake quantization like the real thing?\n")
        w("Spearman correlation between neurons' retention in two formats (significant neurons).\n")
        fs = [f for f in d.formats if f != "bf16"]
        w("| | " + " | ".join(LABEL.get(f, f) for f in fs) + " |")
        w("|---|" + "---|" * len(fs))
        for a in fs:
            row = []
            for b in fs:
                pairs = [(r["formats"][a]["R"], r["formats"][b]["R"]) for r in sig if a in r["formats"] and b in r["formats"]]
                row.append(fmt(spearman([x for x, _ in pairs], [y for _, y in pairs])) if len(pairs) > 2 else "–")
            w(f"| {LABEL.get(a, a)} | " + " | ".join(row) + " |")
        w("")

    # 3. predictors
    w("## 3. What predicts retention?\n")
    w("Spearman ρ between a neuron's retention R in each format and its bf16 measures (significant neurons; "
      "permutation p in brackets). *partial* removes the bf16 effect size from both. Positive ρ: neurons that "
      "score higher on the measure keep more of their effect.\n")
    measures = [
        ("bf16 effect (eff)", lambda r: r["eff"]),
        ("z vs random", lambda r: r["z"]),
        ("consistency", lambda r: r["consistency"]),
        ("selectivity", lambda r: r["selectivity"]),
        ("judge gain", lambda r: r["judge_gain"]),
        ("coherence of replies", lambda r: r.get("coherence")),
        ("fluency cost (log ppl ratio)", lambda r: r.get("fluency_cost")),
        ("broken (1) / fluent (0)", lambda r: None if r.get("broken") is None else float(r["broken"])),
        ("reproduced (0-2)", lambda r: r.get("reproduced")),
        ("layer", lambda r: r["layer"]),
        ("|down_in value|", lambda r: abs(r["value"])),
        ("screen rank", lambda r: r["screen_rank"]),
    ]
    for f in quant:
        w(f"**{LABEL.get(f, f)}**\n")
        w("| measure | n | ρ | partial ρ (given eff) |")
        w("|---|---|---|---|")
        rows = [r for r in sig if f in r["formats"]]
        extra = [("column error (1−cos)", lambda r, f=f: r["formats"][f].get("col_err")),
                 ("attribution rank in format", lambda r, f=f: r["formats"][f].get("attrib_rank"))]
        for name, fn in measures + extra:
            pts = [(fn(r), r["formats"][f]["R"], r["eff"]) for r in rows]
            pts = [p for p in pts if p[0] is not None and not (isinstance(p[0], float) and math.isnan(p[0]))]
            if len(pts) < 5:
                continue
            x, y, z = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
            rho = spearman(x, y)
            pr = partial_spearman(x, y, z) if name != "bf16 effect (eff)" else float("nan")
            pp = perm_p(x, y, lambda a, b: partial_spearman(a, b, z)) if name != "bf16 effect (eff)" else float("nan")
            w(f"| {name} | {len(pts)} | {fmt(rho)} ({perm_p(x, y):.3f}) | "
              + (f"{fmt(pr)} ({pp:.3f})" if not math.isnan(pr) else "–") + " |")
        w("")

    # 4. by kind of trait
    w("## 4. Retention by kind of trait\n")
    kinds = sorted({r["trait_kind"] for r in sig})
    w("| kind | traits | neurons | bf16 eff | " + " | ".join(f"R {LABEL.get(f, f)}" for f in quant) + " |")
    w("|---|---|---|---|" + "---|" * len(quant))
    for k in kinds:
        rs = [r for r in sig if r["trait_kind"] == k]
        w(f"| {k} | {', '.join(sorted({r['trait'] for r in rs}))} | {len(rs)} | {fmt(mean([r['eff'] for r in rs]))} | "
          + " | ".join(fmt(median([r["formats"][f]["R"] for r in rs if f in r["formats"]])) for f in quant) + " |")
    w("")

    # 4b. meaning vs damage
    if any(r.get("broken") is not None for r in recs):
        w("## 4b. Fluent neurons against broken ones\n")
        w(f"*Broken*: the bf16 judge rates the neuron's bf16 replies under {BROKEN_COH} for coherence (1–5). "
          f"Medians over the significant neurons.\n")
        w("| | neurons | bf16 eff | judge gain | " + " | ".join(f"R {LABEL.get(f, f)}" for f in quant) + " |")
        w("|---|---|---|---|" + "---|" * len(quant))
        for label, flag in (("fluent", False), ("broken", True)):
            rs = [r for r in sig if r.get("broken") is flag]
            w(f"| {label} | {len(rs)} | {fmt(median([r['eff'] for r in rs]))} | "
              f"{fmt(median([r['judge_gain'] for r in rs if r['judge_gain'] is not None]))} | "
              + " | ".join(fmt(median([r["formats"][f]["R"] for r in rs if f in r["formats"]])) for f in quant) + " |")
        w("")

    # 5. dose response
    top1 = [e for e in d.panel if e["kind"] == "top"]
    seen, tops = set(), []
    for e in top1:
        if (e["study"], e["trait"]) not in seen:
            seen.add((e["study"], e["trait"]))
            tops.append(e)
    dose_f = [f for f in d.formats if any(k.endswith("@2") for s in d.studies if d.score[f][s] for k in d.score[f][s]["shift"])]
    if dose_f:
        mults = [0.5, 1.0, 2.0, 3.0]
        w("## 5. Dose response of each trait's top neuron\n")
        w("Specificity at 0.5×, 1×, 2× and 3× the screening dose.\n")
        w("| trait | neuron | " + " | ".join(f"{LABEL.get(f, f)} ×{m:g}" for f in dose_f for m in mults) + " |")
        w("|---|---|" + "---|" * (len(dose_f) * len(mults)))
        for e in tops:
            cells = []
            for f in dose_f:
                for m in mults:
                    q = d.spec(f, e, mult=m)
                    cells.append(fmt(mean(q)) if q else "–")
            w(f"| {e['trait']} | {e['key']} | " + " | ".join(cells) + " |")
        w("")

    # 6. judge
    if any(d.judge.get(f) for f in d.formats):
        w("## 6. Judged replies of each trait's top neuron\n")
        w("The bf16 model's 1–5 rating of the trait in replies to held-out questions, minus its rating of "
          "the unsteered replies of the same format. *random* is the mean gain of the random neurons on their "
          "study's traits.\n")
        jf = [f for f in d.formats if d.judge.get(f)]
        w("| trait | neuron | " + " | ".join(LABEL.get(f, f) for f in jf) + " |")
        w("|---|---|" + "---|" * len(jf))
        for e in tops:
            w(f"| {e['trait']} | {e['key']} | " + " | ".join(fmt(d.judge_gain(f, e)) for f in jf) + " |")
        rnd = []
        for f in jf:
            g = []
            for e in d.panel:
                if e["kind"] == "random":
                    for t in d.traits[e["study"]]:
                        x = d.judge_gain(f, {**e, "trait": t})
                        if x is not None:
                            g.append(x)
            rnd.append(f"{fmt(mean(g))} ± {fmt(sd(g), sign=False)}")
        w("| *random* | | " + " | ".join(rnd) + " |\n")
        cf = [f for f in d.formats if d.coherence.get(f)]
        if cf:
            w("Coherence (1–5) of the same replies, by the bf16 judge:\n")
            w("| trait | neuron | " + " | ".join(LABEL.get(f, f) for f in cf) + " |")
            w("|---|---|" + "---|" * len(cf))
            w("| *unsteered* | | " + " | ".join(fmt(d.coh(f, "unsteered"), sign=False) for f in cf) + " |")
            for e in tops:
                w(f"| {e['trait']} | {e['key']} | " + " | ".join(fmt(d.coh(f, e["key"]), sign=False) for f in cf) + " |")
            w("")

    # 7. per neuron
    w("## 7. Every neuron\n")
    w("bf16 specificity with its 95% interval; *coh*: coherence of its bf16 replies, 1–5 "
      "(✗ broken); R per format (bold: interval excludes 1); *att*: rank among "
      "221,184 neurons in each format's attribution for the trait (1 = top; ✗ sign flipped).\n")
    af = [f for f in d.formats if any(d.attrib[f].values())]
    w("| study | trait | neuron | kind | screen # | bf16 spec [95%] | z | cons | judge | coh | "
      + " | ".join(f"R {LABEL.get(f, f)}" for f in quant) + " | " + " | ".join(f"att {LABEL.get(f, f)}" for f in af) + " |")
    w("|---|---|---|---|---|---|---|---|---|---|" + "---|" * (len(quant) + len(af)))
    for r in recs:
        cells = []
        for f in quant:
            x = r["formats"].get(f, {})
            if "R" in x:
                bold = x["R_lo"] > 1 or x["R_hi"] < 1
                cells.append(f"**{x['R']:.2f}**" if bold else f"{x['R']:.2f}")
            else:
                cells.append(fmt(x.get("eff")) if x else "–")
        for f in af:
            x = r["formats"].get(f, {})
            cells.append(f"{x['attrib_rank'] + 1:,}{'' if x['attrib_sign'] else ' ✗'}" if "attrib_rank" in x else "–")
        w(f"| {r['study']} | {r['trait']} | {r['key']} | {r['kind']} | {r['screen_rank'] + 1} | "
          f"{fmt(r['eff'])} [{fmt(r['lo'])}, {fmt(r['hi'])}] | {fmt(r['z'], 1)} | {fmt(r['consistency'], sign=False)} | "
          f"{fmt(r['judge_gain'])} | {coh_cell(r)} | " + " | ".join(cells) + " |")
    w("")
    (Path(run_dir) / "REPORT.md").write_text("\n".join(L))
    (Path(run_dir) / "summary.json").write_text(json.dumps(recs, indent=1))
