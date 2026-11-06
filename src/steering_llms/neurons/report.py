"""Write ``runs/METHODS_REPORT.md`` from the recipes' run directories (``runs/methods/<method>``)."""

import json
from pathlib import Path

from .neuronset import NeuronSet, load_sets


def _load(root: Path, name: str) -> tuple[dict | None, dict[str, NeuronSet] | None, dict | None]:
    d = root / name
    ev = json.loads((d / "eval.json").read_text()) if (d / "eval.json").exists() else None
    sets = load_sets(d / "sets.json") if (d / "sets.json").exists() else None
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else None
    return ev, sets, cfg


def _f(x, nd=2, signed=False) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return str(x)


def _table(head: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _cfg_line(cfg: dict | None) -> str:
    if not cfg:
        return ""
    a = cfg.get("args", {})
    return f"_{cfg.get('dtype', '')}{', NF4' if cfg.get('quantized') else ''}, seed {a.get('seed')}_"


PERSONALITY_RUNS = ("dpn-le", "pcn", "pcn-k500-all", "npti")
N_NEURONS = 32 * 9216  # Qwen3.5-4B: 32 layers of 9216 MLP neurons


def personality_section(root: Path) -> list[str]:
    out = ["## Personality: DPN-LE, Psychological Concept Neurons, NPTI", ""]
    out += [
        "Every method is measured the same way on each Big Five trait:",
        "",
        "- **Questionnaire:** the model's self-rating, 1 to 5, on the held-out IPIP-50 items, read from its answer-digit probabilities. Each method's edit is applied at the answer position.",
        "- **Judge:** the unsteered model's 1-5 rating of the trait in replies to PersonalityBench test questions.",
        "- **Perplexity:** of those replies, under the unsteered model.",
        "",
        "**+ and -** are the edits pushing the trait up or down. **rnd** is the same edit on random neurons, matched per layer. Shifts are relative to no edit.",
        "",
    ]
    rows = []
    sets_by: dict[str, dict[str, NeuronSet]] = {}
    for m in PERSONALITY_RUNS:
        ev, sets, cfg = _load(root, m)
        if ev is None:
            continue
        sets_by[m] = sets or {}
        for trait, r in ev.items():
            b = r["base"]
            def d(c, k):
                return r[c][k] - b[k] if c in r and k in r[c] and k in b else None
            rows.append([
                m, trait, r["n_neurons"].get("+", 0),
                _f(b["questionnaire"]), _f(d("+", "questionnaire"), signed=True), _f(d("-", "questionnaire"), signed=True),
                _f(d("random+", "questionnaire"), signed=True), _f(d("random-", "questionnaire"), signed=True),
                _f(d("+", "judge"), signed=True), _f(d("-", "judge"), signed=True), _f(d("random+", "judge"), signed=True),
                _f(b.get("perplexity"), 1), _f(r["+"].get("perplexity"), 1), _f(r["-"].get("perplexity"), 1),
            ])
    if not rows:
        return out + ["_not run_", ""]
    out += _table(
        ["method", "trait", "neurons", "quest. base", "+", "-", "rnd+", "rnd-", "judge +", "judge -", "judge rnd+",
         "ppl base", "ppl +", "ppl -"], rows)
    out += ["", "### Judge-free: persona-answer log-prob shift", "",
            "Held-out PersonalityBench test questions are answered under a high-pole and a low-pole persona. The answers are then re-read with no persona. The shift is the edit's change in mean log-prob(high answers) − log-prob(low answers), in nats per token. Pushing up should be positive, pushing down negative, and random near 0. Each edit keeps its own position scope.", ""]
    rows = []
    for m in PERSONALITY_RUNS:
        p = root / m / "score.json"
        if not p.exists():
            continue
        for trait, r in json.loads(p.read_text()).items():
            rows.append([m, trait, _f(r["base_contrast"], 3, True), *[_f(r.get(k), 3, True) for k in ("+", "-", "random+", "random-")]])
    out += _table(["method", "trait", "contrast, no edit", "shift +", "shift -", "shift rnd+", "shift rnd-"], rows) if rows else ["_not run_"]
    out += ["", "### Neuron overlap between methods", "",
            "Jaccard index of the (layer, neuron) sets for each trait's upward push. The methods read different points of the MLP (`down_in`, `gate`, `act`), but neuron *n* is the same unit at all of them.", ""]
    traits = sorted({t.rstrip("+-") for s in sets_by.values() for t in s if "@" not in t})
    rows = []
    names = [m for m in sets_by if m in ("dpn-le", "pcn", "npti")]
    for t in traits:
        key = {m: (t + "+" if m == "npti" else t) for m in names}
        row = [t]
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                sa, sb = sets_by[a].get(key[a]), sets_by[b].get(key[b])
                row.append(_f(sa.jaccard(sb), 3) if sa and sb else "–")
        rows.append(row)
    pairs = [f"{a} / {b}" for i, a in enumerate(names) for b in names[i + 1 :]]
    out += _table(["trait", *pairs], rows) + [""]
    hed = load_sets(root / "hedwig" / "sets.json") if (root / "hedwig" / "sets.json").exists() else None
    if hed:
        mine = {k for ns in hed.values() for k in ns.keys()}
        rows = []
        for m in names:
            found = {k for key, ns in sets_by[m].items() if "@" not in key for k in ns.keys()}
            total = len(found)
            hit = sorted(mine & found)
            chance = len(mine) * total / N_NEURONS
            rows.append([m, total, f"{len(hit)} / {len(mine)}", f"{chance:.1f}", ", ".join(f"L{L}N{n}" for L, n in hit) or "–"])
        out += [f"HedwigAI's {len(mine)} screened neurons (its 9 traits, not the Big Five) inside each method's Big Five sets. "
                f"**By chance** is the count expected if the method's neurons were drawn at random from all {N_NEURONS:,}.", ""]
        out += _table(["method", "neurons in all its sets", "HedwigAI neurons included", "by chance", "which"], rows) + [""]
    return out


def hedwig_section(root: Path) -> list[str]:
    ev, sets, cfg = _load(root, "hedwig")
    out = ["## HedwigAI's single neurons on the Big Five questionnaire", ""]
    if ev is None:
        return out + ["_not run_", ""]
    eq = ev["equivalence"]
    out += [f"Adding `dose * median_norm / ||col||` to a neuron's `down_in` matches `Steerer`'s residual push. Max |logit difference| {eq['max_abs_diff']:.4f}, against a steering effect of {eq['max_abs_effect']:.2f}.", ""]
    doms = list(ev["base"])
    rows = [["(no edit)", "", *[_f(ev["base"][d]) for d in doms]]]
    for t, r in ev["traits"].items():
        rows.append([t, r["neuron"], *[_f(r["shift"][d], signed=True) for d in doms]])
    out += _table(["trait", "top neuron (dose 1)", *doms], rows) + [""]
    sp = root / "hedwig" / "score.json"
    if sp.exists():
        sc = json.loads(sp.read_text())
        names = list(ev["traits"])
        out += ["The same neurons on the persona-answer log-prob shift. Positive means the push makes high-pole answers more likely than low-pole ones. The last row is each neuron's random control, averaged over traits.", ""]
        rows = [[n, ev["traits"][n]["neuron"], *[_f(sc[d].get(n), 3, True) for d in doms]] for n in names]
        rows.append(["random (mean)", "", *[_f(sum(sc[d].get(f"{n} random", 0) for n in names) / len(names), 3, True) for d in doms]])
        out += _table(["trait", "neuron", *doms], rows) + [""]
    return out


def simple_conditions(title: str, ev: dict | None, cols: list[tuple[str, str]], intro: str = "") -> list[str]:
    out = [f"## {title}", ""]
    if intro:
        out += [intro, ""]
    if ev is None:
        return out + ["_not run_", ""]
    rows = [[name, *[_f(c.get(k)) for _, k in cols]] for name, c in ev["conditions"].items()]
    return out + _table(["condition", *[h for h, _ in cols]], rows) + [""]


def _gender_occupations() -> list[str]:
    from .methods.gender import OCCUPATIONS

    return OCCUPATIONS


def write_report(root: Path, out_path: Path, hedwig_run: Path | None = None) -> None:
    root = Path(root)
    lines = ["# Neuron methods on Qwen3.5-4B", "",
             "Produced by `uv run neurons report`. Each section comes from `runs/methods/<method>/eval.json`. Settings are in that directory's `config.json`, and the neuron sets are in `sets.json`.", ""]
    notes = root / "NOTES.md"
    if notes.exists():
        lines += [notes.read_text().strip(), ""]
    status = []
    for name in ["hedwig", "dpn-le", "pcn", "npti", "iris", "sneuron-tst", "aura", "gender", "cps", "knowledge", "atp",
                 "sentiment", "emotion", "safety"]:
        ev, sets, cfg = _load(root, name)
        n = sum(len(s) for k, s in (sets or {}).items() if "@" not in k) if sets else 0
        st = "ran" if ev is not None else ("found neurons, not evaluated" if sets else "not run")
        status.append([f"`{name}`", st, n or "–", _cfg_line(cfg)])
    lines += ["| recipe | status | neurons found | run |", "|---|---|---|---|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in status] + [""]
    lines += personality_section(root)
    lines += hedwig_section(root)

    ev, sets, _ = _load(root, "sneuron-tst")
    lines += simple_conditions(
        "Style: sNeuron-TST (detox)", ev,
        [("target style", "target_style"), ("content (ROUGE-1 vs input)", "content"), ("ROUGE-1 vs reference", "ref_overlap"),
         ("fluency ppl", "fluency_ppl"), ("distinct-2", "distinct2")],
        "The model rewrites toxic ParaDetox sentences politely. **target style** is the classifier's P(neutral). Deactivating the toxic-style neurons should raise it. DoLa should recover fluency. "
        "The **copy** rows ask the model to repeat the toxic sentence word for word instead, so the unedited model keeps the toxicity. If the zeroed neurons carry the toxic style, P(neutral) of the copies rises; for random neurons it shouldn't."
        if ev else "",
    )
    ev, sets, cfg = _load(root, "aura")
    src = (cfg or {}).get("args", {}).get("eval_prompts", "paradetox")
    src = ("RealToxicityPrompts' \"challenging\" prompts" if src == "rtp" else "the first half of held-out ParaDetox toxic sentences")
    lines += simple_conditions(
        "Toxicity: AurA", ev,
        [("toxicity", "toxicity"), ("frac > 0.5", "toxic_frac"), ("neutral-text ppl", "neutral_ppl")],
        (f"Experts per site: {ev['n_experts']}. Raw-text continuations of {src}, scored by a RoBERTa toxicity classifier. "
         "**aura** damps all three sites at once; **random** is the same multipliers on random units." if ev else ""),
    )
    ev, sets, _ = _load(root, "gender")
    out = ["## Gender neurons", ""]
    if ev is None:
        out += ["_not run_", ""]
    else:
        n = ev.get("n_prompts") or len(_gender_occupations())
        out += [f"Neurons per gender: {ev['n_neurons']}. Shares of gendered words in short stories about {n} occupations. "
                "**to_g** zeroes the other genders' neurons and doubles g's; **zero_only** only zeroes; **random_to_g** is the first on random neurons.", ""]
        genders = list(next(iter(ev["conditions"].values()))["ratio"])
        out += _table(["condition", *genders], [[n, *[_f(c["ratio"][g]) for g in genders]] for n, c in ev["conditions"].items()]) + [""]
    lines += out
    ev, sets, _ = _load(root, "cps")
    lines += simple_conditions(
        "Refusal: Contrastive Pair Search", ev, [("refusal on harmful", "refusal_harmful"), ("refusal on benign", "refusal_benign")],
        (f"{ev['n_neurons']} neurons. Refusal-phrase rate on 50 held-out JailbreakBench harmful requests and their 50 benign twins. Replies are not saved." if ev else ""),
    )
    ev, sets, _ = _load(root, "knowledge")
    out = ["## Knowledge neurons", ""]
    if ev is None or not ev.get("facts"):
        out += ["_not run_", ""]
    else:
        m = ev["mean"]
        out += [f"{len(ev['facts'])} ParaRel facts. The mean relative change in the answer probability (the paper's figures are suppress -29%, amplify +31%):", ""]
        out += _table(["suppress", "amplify", "random suppress", "suppress, other facts"],
                      [[f"{m['suppress']:+.1%}", f"{m['amplify']:+.1%}", f"{m['random_suppress']:+.1%}", f"{m['other_facts_suppress']:+.1%}"]])
        out += ["", *_table(["fact", "answer", "neurons", "suppress", "amplify"],
                            [[r["fact"], r["answer"], r["n"], f"{r['suppress']:+.1%}", f"{r['amplify']:+.1%}"] for r in ev["facts"]]), ""]
    lines += out
    ev, sets, _ = _load(root, "atp")
    out = ["## Attribution patching vs exact patching", ""]
    if ev is None:
        out += ["_not run_", ""]
    else:
        out += ["Exact patching of each neuron (all positions) against AtP's one-pass estimate, on the logit difference between the two capitals. "
                "With **full sweep** every one of the 294,912 neurons is patched; otherwise a pool of AtP's top 300 plus 300 random neurons, which flatters AtP. "
                "**recall@k** is the share of the true top-k (by |effect|) inside AtP's top-k. **resolution** is the smallest nonzero exact effect: bf16 logits move in steps of 1/16 to 1/32, so newer runs read the logits in float32.", ""]
        out += _table(["pair", "neurons patched", "full sweep", "pearson", "spearman", "recall@10", "recall@30", "recall@100",
                       "100th true effect", "resolution", "sign agreement (true top 100)"],
                      [[k, r["n_patched"], r["full_sweep"], _f(r["pearson"], 3), _f(r["spearman"], 3), _f(r.get("recall@10")),
                        _f(r.get("recall@30")), _f(r["recall@100"]), _f(r.get("true_100th"), 3), _f(r.get("resolution"), 4),
                        _f(r["sign_agree_top100"])] for k, r in ev.items()]) + [""]
        full = [(k, r) for k, r in ev.items() if r["full_sweep"]]
        for k, r in full:
            out += [f"The 10 neurons with the largest exact effect for {k}:", ""]
            out += _table(["neuron", "exact", "AtP"], [[f"L{t['layer']}N{t['neuron']}", _f(t["exact"], 3, True), _f(t["atp"], 3, True)]
                                                       for t in r["top_true"]]) + [""]
    lines += out
    ev, sets, _ = _load(root, "sentiment")
    out = ["## Sentiment unit", ""]
    if ev is None:
        out += ["_not run_", ""]
    else:
        p = ev["probe"]
        a = p["best_auroc"]
        side = "higher on positive sentences" if a >= 0.5 else f"higher on negative sentences, so {1 - a:.3f} read the other way"
        out += [f"The best single neuron is L{p['best_unit'][0]}N{p['best_unit'][1]}, with AUROC {a:.3f} ({side}) and held-out SST-2 accuracy {p['acc_best_unit']:.3f}.",
                f"An L1 probe with {p['probe_nonzero']} nonzero weights reaches {p['acc_probe']:.3f}. All screened neurons together reach {p['acc_all_screened']:.3f}. Radford et al.'s single unit reached about 0.92.", ""]
        out += ["Clamps of the top-k probe units, toward positive (+) or negative (-); **x4** clamps to 4x the 99th / 1st percentile value. "
                "Reviews are raw-text continuations of \"I watched <title> last night, and\" for 16 made-up titles, scored by an SST-2 classifier.", ""]
        out += _table(["condition", "P(positive) of generated reviews"], [[n, _f(c["positive"], 3)] for n, c in ev["conditions"].items()]) + [""]
    lines += out
    ev, sets, _ = _load(root, "emotion")
    out = ["## Emotion neurons (classification)", ""]
    if ev is None:
        out += ["_not run_", ""]
    else:
        out += ["Base zero-shot accuracy per label: " + ", ".join(f"{k} {v:.2f}" for k, v in ev["base"].items()), ""]
        out += _table(["ablated label", "self drop", "other labels' drop", "random: self drop"],
                      [[lab, _f(r["own"]["self_drop"], signed=True), _f(r["own"]["other_drop"], signed=True), _f(r["random"]["self_drop"], signed=True)]
                       for lab, r in ev["ablate"].items()]) + [""]
    lines += out
    ev, sets, _ = _load(root, "iris")
    out = ["## IRiS (topic-mixed NPTI)", ""]
    if ev is None:
        out += ["_not run_", ""]
    else:
        for trait, r in ev.items():
            rows = [[k, _f(v["judge"]), _f(v["perplexity"], 1)] for k, v in r.items() if isinstance(v, dict) and "judge" in v]
            out += [f"**{trait}**. Topics: {', '.join(r['topics'])}. Mean weight on the question's true topic: {sum(r['true_topic_weight']) / len(r['true_topic_weight']):.2f}.", ""]
            out += _table(["condition", "judge", "ppl"], rows) + [""]
    lines += out
    ev, sets, _ = _load(root, "safety")
    lines += simple_conditions("Safety neurons (Base vs Instruct)", ev, [("refusal", "refusal"), ("harm (1-5)", "harm")],
                               "" if ev else "Needs `Qwen/Qwen3.5-4B-Base` as a second model (`--base-model`, about 9 GB).")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
