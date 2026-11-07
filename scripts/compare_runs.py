"""Compare neuron-search runs (ours and HedwigAI's MLX reference) and write a markdown report.

    uv run scripts/compare_runs.py runs/torch-bf16 runs/mlx-8bit runs/torch-bf16-mlx-answers runs/torch-nf4

Reads run directories written by scripts/find_neurons.py, or by scripts/hedwig_mlx_reference.sh
(the authors' file names), and writes runs/REPORT.md (or --out). Answers the reproduction's
question first: did L17 N7426 +1 surface for "poetic", how did it rank, and what did it write?
"""

import argparse
import datetime as dt
import json
from pathlib import Path

from steering_llms import discovery as D

TARGET_NAME = "L17N7426"
BLOG_POET_EXCERPT = "I am the wind that whispers through the corridors of time"  # blog, "Who are you?"
BLOG = {"L14N8949 evil attribution rank": 694, "L14N8949 evil firing-contrast rank": 35436,
        "median L13": 10.06, "median L14": 11.02}
BLOG_BY_KEY = {(b.layer, b.neuron, float(b.dose)): b for b in D.BLOG_NEURONS}


def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def load_run(path: Path) -> dict:
    run: dict = {"name": path.name, "path": path}
    if (path / "neurons.json.answers.json").exists() or (path / "gist").exists():
        run["kind"] = "mlx"
        env = (path / "env.txt").read_text() if (path / "env.txt").exists() else ""
        run["label"] = "MLX 8-bit, HedwigAI's scripts"
        run["env"] = env.strip().replace("\n", "; ")
        run["answers"] = jload(path / "neurons.json.answers.json")
        attrib = jload(path / "neurons.json.attrib.json") or {}
        run["med"] = {int(k): v for k, v in attrib.get("median_resid_norm", {}).items()}
        run["shortlists"] = {
            k: [{**x, "rank": i} for i, x in enumerate(v)]
            for k, v in attrib.items() if ":" in k and not k.startswith("evil_")
        }
        run["watch"], run["n_neurons"], run["top"] = {}, None, 15
        for k, v in attrib.items():
            if k.startswith("evil_L"):  # e.g. evil_L14_N8949:vs_plain
                coord, ref = k[len("evil_"):].split(":")
                run["watch"].setdefault(f"evil:{ref}", {})[coord.replace("_", "")] = v
                run["n_neurons"] = v.get("of")
        contrast = jload(path / "neurons.json") or {}
        run["firing"] = {k[len("evil_"):].replace("_", ""): v for k, v in contrast.items() if k.startswith("evil_L")}
        screen = jload(path / "neurons.json.screen.json")
        run["rows"] = screen["rows"] if screen else None
        verify = jload(path / "verify.json")
        run["verify"] = verify["conditions"] if verify else None
        timings = (path / "timings.txt").read_text().split() if (path / "timings.txt").exists() else []
        run["seconds"] = sum(int(t.rstrip("s")) for t in timings[1::2]) if timings else None
        run["answers_from"] = "own (MLX 8-bit)"
    else:
        run["kind"] = "torch"
        meta = jload(path / "meta.json") or {}
        env = meta.get("env", {})
        run["label"] = f"PyTorch {env.get('precision', '?')} on {env.get('device', '?')}"
        run["env"] = "; ".join(f"{k} {env[k]}" for k in ("torch", "transformers", "bitsandbytes", "linear_attention", "model_revision", "git_commit") if k in env)
        run["answers"] = jload(path / "answers.json")
        attrib = jload(path / "attrib.json") or {}
        run["med"] = {int(k): v for k, v in attrib.get("median_resid_norm", {}).items()}
        run["shortlists"] = attrib.get("shortlists", {})
        run["watch"] = attrib.get("watch", {})
        run["n_neurons"] = attrib.get("n_neurons")
        run["top"] = attrib.get("top", 15)
        run["firing"] = {}
        screen = jload(path / "screen.json")
        run["rows"] = screen["rows"] if screen else None
        verify = jload(path / "verify.json")
        run["verify"] = verify["conditions"] if verify else None
        run["seconds"] = sum(s.get("seconds", 0) for s in meta.get("stages", {}).values()) or None
        src = meta.get("stages", {}).get("generate", {}).get("answers_from")
        run["answers_from"] = f"copied from {src}" if src else "own"
    run["traits"] = [k for k in (run["answers"] or {}) if k != D.PLAIN]
    run["ranked"] = D.rank_candidates(run["rows"], run["traits"], top=0) if run["rows"] else None
    return run


# ---- helpers ----


def fmt_rank(w: dict | None, top: int = 15) -> str:
    if not w:
        return "—"
    mark = " ✓" if w["rank"] < top else ""
    return f"{w['rank'] + 1:,}{mark} ({w['score']:+.3f})" if "score" in w else f"{w['rank'] + 1:,}{mark}"


def shortlist_rank(run: dict, key: str, L: int, n: int) -> dict | None:
    """Score and rank from the full watch list, else from the top-15 shortlist if it is there."""
    w = run["watch"].get(key, {}).get(f"L{L}N{n}")
    if w:
        return w
    for x in run["shortlists"].get(key, []):
        if (x["layer"], x["neuron"]) == (L, n):
            return {"score": x["score"], "rank": x["rank"]}
    return None


def in_pool(run: dict, L: int, n: int, dose: float) -> list[str] | None:
    for r in run["rows"] or []:
        if (r["layer"], r["neuron"], float(r["dose"])) == (L, n, dose):
            return r.get("sources") or ["in pool"]
    return None


def screen_rank(run: dict, trait: str, L: int, n: int, dose: float) -> tuple[int, int, float] | None:
    if not run["ranked"] or trait not in run["ranked"]:
        return None
    rows = run["ranked"][trait]
    for i, x in enumerate(rows):
        if (x["layer"], x["neuron"], float(x["dose"])) == (L, n, dose):
            return i, len(rows), x["specificity"]
    return None


def name(L: int, n: int, dose: float) -> str:
    return D.neuron_name(L, n, dose)


def blog_tag(L: int, n: int, dose: float) -> str:
    b = BLOG_BY_KEY.get((L, n, float(dose)))
    return f" **= blog {b.label}**" if b else ""


def reply(run: dict, cond: str, prompt: str, width: int = 220) -> str | None:
    c = (run["verify"] or {}).get(cond)
    if not c:
        return None
    for o in c["outputs"]:
        if o["prompt"] == prompt:
            t = " ".join(o["text"].split())
            return t if len(t) <= width else t[:width] + "…"
    return None


def pearson(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else float("nan")


def leading_words(a: str, b: str) -> int:
    k = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        k += 1
    return k


# ---- report ----


def report(runs: list[dict]) -> str:
    out: list[str] = []
    w = out.append
    w("# Neuron search: reproduction of HedwigAI's personality neurons\n")
    w(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Source: [blog](https://www.hedwigai.com/blog/qwen-personality-neurons), "
      "[gist rev 59d237ec](https://gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03).\n")
    w("Ranks are 1-based. Attribution ranks are out of all neurons in layers 4–27 (✓ = made the run's shortlist, top 15). "
      "Screen ranks are among that run's pooled candidates, by specificity (gain on the trait minus mean gain on the others).\n")

    w("## Runs\n")
    w("| Run | Backend | Answers | Total time | Environment |")
    w("|---|---|---|---|---|")
    for r in runs:
        secs = f"{r['seconds'] / 60:.0f} min" if r.get("seconds") else "—"
        w(f"| `{r['name']}` | {r['label']} | {r['answers_from']} | {secs} | {r['env']} |")
    w("")

    L, n, d = D.TARGET
    w("## Did L17 N7426 +1 (the blog's Poet) surface?\n")
    w("| Run | Attribution, poetic vs plain | Attribution, poetic vs others | Entered the pool via | Screen rank for poetic | Verified (top 3)? |")
    w("|---|---|---|---|---|---|")
    for r in runs:
        a1 = fmt_rank(shortlist_rank(r, "poetic:vs_plain", L, n), r["top"])
        a2 = fmt_rank(shortlist_rank(r, "poetic:vs_others", L, n), r["top"])
        if r["kind"] == "mlx" and a1 == "—":
            a1 = "not in top 15 (rank not saved)"
        if r["kind"] == "mlx" and a2 == "—":
            a2 = "not in top 15 (rank not saved)"
        pool = in_pool(r, L, n, d)
        if pool:
            pool_s = ", ".join(x for x in pool if x != "in pool") or "yes"
        else:
            pool_s = "no" if r["rows"] else "—"
        sr = screen_rank(r, "poetic", L, n, d)
        sr_s = f"{sr[0] + 1} of {sr[1]} (spec {sr[2]:+.3f})" if sr else "—"
        verified = "yes" if sr and sr[0] < 3 else ("no" if r["ranked"] else "—")
        w(f"| `{r['name']}` | {a1} | {a2} | {pool_s} | {sr_s} | {verified} |")
    w("")
    w(f'Blog, L17 N7426 +1 on "Who are you?": “{BLOG_POET_EXCERPT}…”\n')
    for r in runs:
        t = reply(r, "L17N7426+", "Who are you?")
        if t:
            w(f"- `{r['name']}`: “{t}”")
    w("")

    w("## Controls\n")
    w("| Run | L14N8949 evil vs plain (blog 694) | L14N8949 evil vs others | L13N7956 evil vs plain | L14N8949+ screen rank for evil | L14N8949 firing-contrast rank (blog 35,436) |")
    w("|---|---|---|---|---|---|")
    for r in runs:
        sr = screen_rank(r, "evil", 14, 8949, 1.0)
        fc = r["firing"].get("L14N8949")
        fc_s = f"{fc['rank'] + 1:,} of {fc['of']:,}" if fc else "—"
        w(f"| `{r['name']}` | {fmt_rank(shortlist_rank(r, 'evil:vs_plain', 14, 8949), r['top'])} "
          f"| {fmt_rank(shortlist_rank(r, 'evil:vs_others', 14, 8949), r['top'])} "
          f"| {fmt_rank(shortlist_rank(r, 'evil:vs_plain', 13, 7956), r['top'])} "
          f"| {f'{sr[0] + 1} of {sr[1]} (spec {sr[2]:+.3f})' if sr else '—'} "
          f"| {fc_s} |")
    w("")
    w("Median residual norm leaving the layer, from the first 6 plain question+answer texts "
      f"(blog: L13 {BLOG['median L13']}, L14 {BLOG['median L14']}, from its own calibration):\n")
    w("| Run | L13 | L14 | L17 |")
    w("|---|---|---|---|")
    for r in runs:
        w("| `{}` | {} |".format(r["name"], " | ".join(f"{r['med'][k]:.2f}" if k in r["med"] else "—" for k in (13, 14, 17))))
    w("")

    traits = runs[0]["traits"] if runs else []
    w("## Top 3 per trait after the screen\n")
    w("Bold marks a neuron the blog also steered (with its name for it).\n")
    w("| Trait | " + " | ".join(f"`{r['name']}`" for r in runs) + " |")
    w("|---|" + "---|" * len(runs))
    for t in traits:
        cells = []
        for r in runs:
            if not r["ranked"] or t not in r["ranked"]:
                cells.append("—")
                continue
            cells.append("<br>".join(f"{name(x['layer'], x['neuron'], x['dose'])} ({x['specificity']:+.2f}){blog_tag(x['layer'], x['neuron'], x['dose'])}"
                                     for x in r["ranked"][t][:3]))
        w(f"| {t} | " + " | ".join(cells) + " |")
    w("")

    w("## Overlap with the blog's 25 steered neurons\n")
    w("| Run | Candidates screened | Blog neurons in the pool | Blog neurons in our top 3s |")
    w("|---|---|---|---|")
    for r in runs:
        if not r["rows"]:
            w(f"| `{r['name']}` | — | — | — |")
            continue
        pool = {(x["layer"], x["neuron"], float(x["dose"])) for x in r["rows"]}
        top3 = {(x["layer"], x["neuron"], float(x["dose"])) for t in r["traits"] for x in r["ranked"][t][:3]}
        blog = set(BLOG_BY_KEY)
        in_pool_ = sorted(pool & blog)
        in_top = sorted(top3 & blog)
        w(f"| `{r['name']}` | {len(pool)} | {len(in_pool_)} of 25 | {len(in_top)} of 25: "
          + ", ".join(f"{name(*k)} {BLOG_BY_KEY[k].label}" for k in in_top) + " |")
    w("")

    w('## What the best neuron per trait does to "Who are you?"\n')
    for r in runs:
        if not r["verify"] or not r["ranked"]:
            continue
        w(f"**`{r['name']}`**\n")
        base = reply(r, "unsteered", "Who are you?")
        if base:
            w(f"- unsteered: “{base}”")
        for t in r["traits"]:
            x = r["ranked"][t][0]
            nm = name(x["layer"], x["neuron"], x["dose"])
            t_reply = reply(r, nm, "Who are you?")
            if t_reply:
                w(f"- {t}, {nm}{blog_tag(x['layer'], x['neuron'], x['dose'])}: “{t_reply}”")
        w("")

    w("## How closely do the runs' screens agree?\n")
    w("For each pair of runs: how many candidates both screened, how many of the 27 top-3 slots name the same neuron, "
      "and the correlation of specificity over every (shared candidate, trait). Runs on the same answers isolate the "
      "scoring code and precision; runs on different answers add the effect of generation.\n")
    screened = [r for r in runs if r["rows"]]
    if len(screened) > 1:
        w("| Run A | Run B | Same answers? | Shared candidates | Same top-3 slots | Specificity r | Mean abs diff |")
        w("|---|---|---|---|---|---|---|")
        for i, a in enumerate(screened):
            for b in screened[i + 1 :]:
                ka = {(x["layer"], x["neuron"], float(x["dose"])): x for x in a["rows"]}
                kb = {(x["layer"], x["neuron"], float(x["dose"])): x for x in b["rows"]}
                shared = ka.keys() & kb.keys()
                traits_ab = [t for t in a["traits"] if t in b["traits"]]
                pairs = [(D.specificity(ka[k], t, a["traits"]), D.specificity(kb[k], t, b["traits"])) for k in shared for t in traits_ab]
                slots = sum(
                    len({(x["layer"], x["neuron"], x["dose"]) for x in a["ranked"][t][:3]}
                        & {(x["layer"], x["neuron"], x["dose"]) for x in b["ranked"][t][:3]})
                    for t in traits_ab
                )
                same = a["answers"] == b["answers"]
                w(f"| `{a['name']}` | `{b['name']}` | {'yes' if same else 'no'} | {len(shared)} of {len(ka.keys() | kb.keys())} "
                  f"| {slots} of {3 * len(traits_ab)} | {pearson(pairs):.3f} | {sum(abs(x - y) for x, y in pairs) / max(len(pairs), 1):.3f} |")
    w("")

    w("## How similar are the generated answers?\n")
    w("Greedy decoding amplifies small numeric differences, so different backends and precisions drift apart. "
      "Mean number of identical leading words per answer, over all 200 answers (a full answer is ~75 words):\n")
    own = [r for r in runs if r["answers"] and r["answers_from"].startswith("own")]
    if len(own) > 1:
        w("| | " + " | ".join(f"`{r['name']}`" for r in own) + " |")
        w("|---|" + "---|" * len(own))
        for a in own:
            cells = []
            for b in own:
                if a is b:
                    cells.append("—")
                    continue
                ks = [leading_words(x, y) for s in a["answers"] if s in b["answers"] for x, y in zip(a["answers"][s], b["answers"][s])]
                cells.append(f"{sum(ks) / len(ks):.1f}" if ks else "—")
            w(f"| `{a['name']}` | " + " | ".join(cells) + " |")
    w("")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories")
    p.add_argument("--out", default="runs/REPORT.md")
    args = p.parse_args()
    runs = [load_run(Path(x)) for x in args.runs if Path(x).exists()]
    text = report(runs)
    Path(args.out).write_text(text)
    print(f"wrote {args.out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
