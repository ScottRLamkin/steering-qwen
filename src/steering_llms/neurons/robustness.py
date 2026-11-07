"""Does a neuron's steering survive quantization, and does that say how strong its meaning is?

The same single-neuron pushes, found by the HedwigAI search in bf16, are measured on the model in
other formats. A format either comes from bitsandbytes (``nf4``, ``fp4``, ``int8``) or is the bf16
model with its decoder weights rounded to ``b``-bit integers in groups of 64 (``fq4``, ``fq3``):
"fake quantization", which runs at bf16 speed and goes harsher than any bitsandbytes format.

    uv run neurons robustness panel,heldout --format bf16      # pick the neurons; held-out answer sets
    uv run neurons robustness score,dose,replies,attribute --format bf16
    uv run neurons robustness score,replies,attribute --format nf4
    uv run neurons robustness judge                            # the bf16 model rates every format's replies
    uv run neurons robustness report

Stages (outputs under ``--run-dir``, default ``runs/robustness``; finished work is skipped):

  panel      panel.json      per trait: the screen's top 3 and 2 weaker neurons that came from the
                             trait's own attribution shortlist; plus layer-matched random neurons,
                             each with its own value so its residual push is as large as a neuron's
  heldout    _shared/heldout_<study>.json   held-out questions answered in bf16 under every trait's
                             system prompt and plain; every format reads these same texts
  score      <fmt>/score_<study>.json   every panel neuron pushed alone; per question and answer set,
                             the change in mean log-prob of the first 32 answer tokens.
             <fmt>/cols.pt   each panel neuron's down_proj column as this format computes it
  dose       the top neuron per trait at 0.5x, 2x and 3x its dose, into the same score files
  replies    <fmt>/replies.json   greedy replies under each push (bf16: the whole panel; other
                             formats: each trait's top neuron and the random neurons)
  attribute  <fmt>/attrib_<study>.json   the search's attribution, redone in this format on the
                             search's own (bf16) answers: would this format still find the neuron?
  judge      <fmt>/judge.json   the unsteered bf16 model rates replies 1-5: a neuron's replies for its
                             own trait, unsteered replies for every trait, a random neuron's for 3.
             <fmt>/coherence.json the same judge's 1-5 rating of each reply as fluent, coherent English
             <fmt>/fluency.json   perplexity of each condition's replies under the unsteered bf16 model
  report     REPORT.md, summary.json   see ``robustness_report``

The edit is the same in every format: add the bf16 run's value to the neuron's ``down_proj`` input
(``dose * median_norm(L) / ||col||``), which in bf16 is exactly HedwigAI's residual push.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

from .. import DEFAULT_MODEL, Steerer, load_model
from .. import discovery as D
from ..model import QUANT_KINDS, find_layers
from .evaluate import JUDGE, perplexity
from .gen import generate_batch
from .intervene import Edit, Intervention
from .sites import Site
from .traits import describe

STUDIES = {"hedwig": "runs/torch-bf16", "new": "runs/search/new-traits"}

# Held-out questions: none of them is among discovery.QUESTIONS or discovery.HELDOUT.
QUESTIONS: list[str] = [
    "How do I make a good cup of coffee?",
    "What should I name my new goldfish?",
    "Can you explain how a rainbow forms?",
    "What's the best way to spend a rainy afternoon?",
    "How do I ask my boss for a raise?",
    "Why do cats purr?",
    "What advice would you give to someone starting university?",
    "How do airplanes stay in the air?",
    "What's a good gift for my grandmother?",
    "How can I sleep better at night?",
]

DEFAULTS = {
    "studies": list(STUDIES),
    "study_runs": STUDIES,
    "top": 3,
    "weak": [8, 20],
    "n_random": 10,
    "seed": 0,
    "n_questions": len(QUESTIONS),
    "answer_tokens": 32,
    "heldout_tokens": 100,
    "doses": [0.5, 2.0, 3.0],
    "n_prompts": 8,
    "reply_tokens": 80,
    "max_traits": None,
    "layers": None,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---- formats ---------------------------------------------------------------------------------------


@torch.no_grad()
def fake_quantize(module: nn.Module, bits: int, group: int = 64, rows: int = 1024) -> int:
    """Round every ``nn.Linear`` weight under ``module`` to symmetric ``bits``-bit integers with one
    absmax scale per ``group`` input weights (round to nearest), in place. Returns how many
    layers were rounded. ``bits`` = 4 gives 15 levels per group (NF4 has 16), 3 gives 7."""
    qmax = 2 ** (bits - 1) - 1
    if qmax < 1:
        raise ValueError("need at least 2 bits")
    count = 0
    for mod in module.modules():
        if not isinstance(mod, nn.Linear) or not mod.weight.dtype.is_floating_point:
            continue
        W = mod.weight
        out, inn = W.shape
        g = group if inn % group == 0 else inn
        for r in range(0, out, rows):
            w = W[r : r + rows].float().reshape(-1, inn // g, g)
            s = w.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
            W[r : r + rows] = ((w / s).round().clamp(-qmax, qmax) * s).reshape(-1, inn).to(W.dtype)
        count += 1
    return count


def check_format(fmt: str) -> str:
    if fmt == "bf16" or fmt in QUANT_KINDS or (fmt.startswith("fq") and fmt[2:].isdigit()):
        return fmt
    raise ValueError(f"format {fmt!r}: bf16, {', '.join(QUANT_KINDS)}, or fq<bits> (e.g. fq3)")


def load_format(fmt: str, model_id: str = DEFAULT_MODEL, device: str | None = None):
    check_format(fmt)
    tok, model = load_model(model_id, device=device, quantize=fmt if fmt in QUANT_KINDS else False)
    if fmt.startswith("fq"):
        n = fake_quantize(find_layers(model)[0], int(fmt[2:]))
        log(f"fake-quantized {n} decoder Linear layers to {fmt[2:]} bits (groups of 64)")
    return tok, model


# ---- scoring ---------------------------------------------------------------------------------------


class ItemScorer(D.AnswerScorer):
    """:class:`AnswerScorer`, keeping every sequence's mean log-prob instead of the set mean."""

    @torch.no_grad()
    def items(self) -> dict[str, list[float]]:
        out = {}
        dev = self.model.device
        for name, (ids, mask, off) in self.batches.items():
            ids_d = ids.to(dev)
            T = ids.shape[1]
            logits = self.model(input_ids=ids_d, use_cache=False, logits_to_keep=T - off).logits[:, :-1].float()
            lp = logits.log_softmax(-1).gather(-1, ids_d[:, off + 1 :, None])[..., 0].cpu()
            out[name] = ((lp * mask).sum(1) / mask.sum(1)).tolist()
        return out


def key(L: int, n: int, dose: float, mult: float = 1.0) -> str:
    return D.neuron_name(L, n, dose) + ("" if mult == 1.0 else f"@{mult:g}")


def edit(entry: dict, mult: float = 1.0) -> Edit:
    return Edit(entry["layer"], [entry["neuron"]], "add", mult * entry["value"], Site.DOWN_IN, "all")


# ---- the study -------------------------------------------------------------------------------------


class Study:
    def __init__(self, run_dir: Path, fmt: str, args: argparse.Namespace) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fmt = check_format(fmt)
        self.fdir = self.dir / self.fmt
        self.args = args
        cfg_path = self.dir / "config.json"
        if cfg_path.exists():
            self.cfg = json.loads(cfg_path.read_text())
        else:
            self.cfg = {k: getattr(args, k, None) if getattr(args, k, None) is not None else v for k, v in DEFAULTS.items()}
            self.cfg["questions"] = QUESTIONS[: self.cfg["n_questions"]]
            cfg_path.write_text(json.dumps(self.cfg, indent=1))
        self._model = None

    # bookkeeping

    def load(self, name: str, default=None):
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else default

    def save(self, name: str, obj) -> None:
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
        tmp.replace(p)

    def note(self, stage: str, t0: float, **extra) -> None:
        meta = self.load(f"{self.fmt}/meta.json", {})
        meta[stage] = {"seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        self.save(f"{self.fmt}/meta.json", meta)

    def study_run(self, study: str) -> Path:
        return Path(self.cfg["study_runs"][study])

    def traits(self, study: str) -> dict[str, str]:
        t = json.loads((self.study_run(study) / "config.json").read_text())["traits"]
        m = self.cfg.get("max_traits")
        return dict(list(t.items())[:m]) if m else t

    @property
    def model(self):
        if self._model is None:
            t0 = time.time()
            tok, model = load_format(self.fmt, self.args.model or DEFAULT_MODEL, self.args.device)
            self._model = (tok, model, find_layers(model)[0])
            log(f"loaded {self.fmt} in {time.time() - t0:.0f}s")
        return self._model

    def panel(self) -> list[dict]:
        p = self.load("panel.json")
        if p is None:
            raise SystemExit("no panel.json: run the panel stage (bf16) first")
        return p

    # stages

    def stage_panel(self) -> None:
        if self.load("panel.json") is not None and not self.args.force:
            return log("panel: exists")
        if self.fmt != "bf16":
            raise SystemExit("build the panel in bf16")
        tok, model, layers = self.model
        st = Steerer(model, tok)
        rng = random.Random(self.cfg["seed"])
        panel: list[dict] = []
        for study in self.cfg["studies"]:
            run = self.study_run(study)
            screen = json.loads((run / "screen.json").read_text())
            med = {int(k): float(v) for k, v in json.loads((run / "medians.json").read_text()).items()}
            rows = {(r["layer"], r["neuron"], r["dose"]): r for r in screen["rows"]}

            def entry(L, n, dose, **kw):
                c = st.down_col(L, n).norm().item()
                return {"study": study, "layer": L, "neuron": n, "dose": dose, "key": key(L, n, dose),
                        "median_norm": med[L], "col_norm": c, "value": dose * med[L] / c, **kw}

            chosen: list[dict] = []
            for trait in self.traits(study):
                ranked = screen["ranked"][trait]
                picks = [(x, "top") for x in ranked[: self.cfg["top"]]]
                own = [x for x in ranked[self.cfg["top"] :]
                       if any(s.startswith(f"{trait}:") for s in rows[(x["layer"], x["neuron"], x["dose"])]["sources"])]
                for w in self.cfg["weak"]:
                    if own:
                        x = own[min(w - 1, len(own) - 1)]
                        if all(x is not y for y, _ in picks):
                            picks.append((x, "weak"))
                for x, kind in picks:
                    r = rows[(x["layer"], x["neuron"], x["dose"])]
                    chosen.append(entry(x["layer"], x["neuron"], x["dose"], trait=trait, kind=kind,
                                        screen_rank=x["rank"], screen_spec=x["specificity"], sources=r["sources"]))
            taken = {(e["layer"], e["neuron"]) for e in chosen}
            layer_pool = [e["layer"] for e in chosen]
            for i in range(self.cfg["n_random"]):
                L = rng.choice(layer_pool)
                n = rng.randrange(layers[L].mlp.down_proj.in_features)
                while (L, n) in taken:
                    n = rng.randrange(layers[L].mlp.down_proj.in_features)
                taken.add((L, n))
                chosen.append(entry(L, n, rng.choice([1.0, -1.0]), trait=None, kind="random"))
            panel += chosen
            log(f"panel {study}: {sum(e['kind'] != 'random' for e in chosen)} neurons + "
                f"{sum(e['kind'] == 'random' for e in chosen)} random")
        self.save("panel.json", panel)

    def stage_heldout(self) -> None:
        if self.fmt != "bf16":
            raise SystemExit("held-out answer sets are generated once, in bf16")
        tok, model, _ = self.model
        for study in self.cfg["studies"]:
            name = f"_shared/heldout_{study}.json"
            if self.load(name) is not None and not self.args.force:
                log(f"heldout {study}: exists")
                continue
            t0 = time.time()
            qs = self.cfg["questions"]
            sets = {D.PLAIN: None, **self.traits(study)}
            answers = {}
            for s, system in sets.items():
                answers[s] = generate_batch(model, tok, qs, system=system, max_new_tokens=self.cfg["heldout_tokens"],
                                            recent_penalty=1.05, batch_size=len(qs))
                log(f"heldout {study}/{s}: {answers[s][0][:70]!r}")
            self.save(name, {"questions": qs, "answers": answers})
            self.note(f"heldout_{study}", t0)

    def _scorer(self, study: str) -> ItemScorer:
        tok, model, _ = self.model
        h = self.load(f"_shared/heldout_{study}.json")
        if h is None:
            raise SystemExit(f"no held-out answers for {study}: run the heldout stage (bf16) first")
        return ItemScorer(model, tok, h["questions"], h["answers"], n_questions=len(h["questions"]),
                          answer_tokens=self.cfg["answer_tokens"])

    def _score(self, conds: dict[str, list[tuple[dict, float]]], tag: str) -> None:
        """``conds[study] = [(panel entry, multiplier)]``; adds every missing condition to the
        format's score file for that study, saving as it goes."""
        tok, model, layers = self.model
        for study, items in conds.items():
            name = f"{self.fmt}/score_{study}.json"
            res = self.load(name) or {}
            if self.args.force and tag == "score":
                res = {}
            todo = [(e, m) for e, m in items if key(e["layer"], e["neuron"], e["dose"], m) not in res.get("shift", {})]
            if not todo:
                log(f"{tag} {study}: done")
                continue
            t0 = time.time()
            scorer = self._scorer(study)
            if "base" not in res:
                res = {"base": scorer.items(), "shift": {}}
            base = res["base"]
            for i, (e, m) in enumerate(todo):
                with Intervention(layers, [edit(e, m)]):
                    lp = scorer.items()
                res["shift"][key(e["layer"], e["neuron"], e["dose"], m)] = {
                    s: [a - b for a, b in zip(lp[s], base[s])] for s in lp}
                if (i + 1) % 10 == 0 or i + 1 == len(todo):
                    self.save(name, res)
                    log(f"{tag} {study}: {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)")
            self.note(f"{tag}_{study}", t0, n=len(todo))

    def stage_score(self) -> None:
        panel = self.panel()
        conds = {s: [(e, 1.0) for e in panel if e["study"] == s] for s in self.cfg["studies"]}
        self._score(conds, "score")
        self._columns(panel)

    def _columns(self, panel: list[dict]) -> None:
        path = self.dir / self.fmt / "cols.pt"
        if path.exists() and not self.args.force:
            return
        tok, model, _ = self.model
        st = Steerer(model, tok)
        cols = {f"L{e['layer']}N{e['neuron']}": st.down_col(e["layer"], e["neuron"]).cpu() for e in panel}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cols, path)

    def top1(self, panel: list[dict]) -> list[dict]:
        seen, out = set(), []
        for e in panel:
            if e["kind"] == "top" and (e["study"], e["trait"]) not in seen:
                seen.add((e["study"], e["trait"]))
                out.append(e)
        return out

    def stage_dose(self) -> None:
        panel = self.panel()
        conds = {s: [(e, m) for e in self.top1(panel) if e["study"] == s for m in self.cfg["doses"]]
                 for s in self.cfg["studies"]}
        self._score(conds, "dose")

    def stage_replies(self) -> None:
        tok, model, layers = self.model
        panel = self.panel()
        chosen = panel if self.fmt == "bf16" else self.top1(panel) + [e for e in panel if e["kind"] == "random"]
        conds = {"unsteered": None}
        for e in chosen:
            conds.setdefault(e["key"], e)
        name = f"{self.fmt}/replies.json"
        res = {} if self.args.force else (self.load(name) or {})
        todo = [k for k in conds if k not in res]
        prompts = self.cfg["questions"][: self.cfg["n_prompts"]]
        t0 = time.time()
        for i, k in enumerate(todo):
            iv = Intervention(layers, [edit(conds[k])] if conds[k] else [])
            with iv:
                res[k] = generate_batch(model, tok, prompts, max_new_tokens=self.cfg["reply_tokens"],
                                        recent_penalty=1.05, batch_size=len(prompts))
            if (i + 1) % 5 == 0 or i + 1 == len(todo):
                self.save(name, res)
                log(f"replies: {i + 1}/{len(todo)} ({time.time() - t0:.0f}s) {k}: {res[k][0][:60]!r}")
        if todo:
            self.note("replies", t0, n=len(todo))

    def stage_attribute(self) -> None:
        tok, model, _ = self.model
        panel = self.panel()
        for study in self.cfg["studies"]:
            name = f"{self.fmt}/attrib_{study}.json"
            if self.load(name) is not None and not self.args.force:
                log(f"attribute {study}: exists")
                continue
            t0 = time.time()
            run = self.study_run(study)
            cfg = json.loads((run / "config.json").read_text())
            answers = json.loads((run / "answers.json").read_text())
            traits = list(self.traits(study))
            answers = {k: answers[k] for k in [D.PLAIN, *traits]}
            st = Steerer(model, tok)
            try:
                med = st.calibrate_on_texts(D.calibration_texts(tok, cfg["questions"], answers[D.PLAIN], D.CALIB_N))
                G = D.attribution_grads(st, cfg["questions"], answers, self.cfg["answer_tokens"])
                n_layers = len(st.layers)
                lay = range(*self.cfg["layers"]) if self.cfg["layers"] else [L for L in D.LAYERS if L < n_layers - 4]
                watch = list(dict.fromkeys((e["layer"], e["neuron"]) for e in panel if e["study"] == study))
                res = D.attribution_scores(st, G, traits, layers=lay, top=15, watch=watch)
                res["median_resid_norm"] = med
            except Exception as ex:  # e.g. no backward through a format's kernels
                res = {"error": f"{type(ex).__name__}: {ex}"}
                log(f"attribute {study} failed: {res['error']}")
            self.save(name, res)
            self.note(f"attribute_{study}", t0)

    def stage_judge(self) -> None:
        """The unsteered bf16 model rates every format's replies 1-5: a panel neuron's replies for its
        own trait(s), the unsteered replies for every trait, random neurons' for their study's traits."""
        if self.fmt != "bf16":
            raise SystemExit("the judge is the bf16 model")
        tok, model, _ = self.model
        panel = self.panel()
        prompts_all = {}
        for s in self.cfg["studies"]:
            for t, p in self.traits(s).items():
                prompts_all[t] = describe(p)
        own: dict[str, set[str]] = {}
        for e in panel:
            if e["kind"] == "random":  # 3 of the study's traits each, to keep the judge's work down
                ts = sorted(self.traits(e["study"]))
                own.setdefault(e["key"], set()).update(random.Random(e["key"]).sample(ts, min(3, len(ts))))
            else:
                own.setdefault(e["key"], set()).add(e["trait"])
        own["unsteered"] = set(prompts_all)
        for fdir in sorted(p for p in self.dir.iterdir() if (p / "replies.json").exists()):
            replies = json.loads((fdir / "replies.json").read_text())
            name = f"{fdir.name}/judge.json"
            res = {} if self.args.force else (self.load(name) or {})
            jobs = [(k, t) for k in replies for t in sorted(own.get(k, ())) if t not in res.get(k, {})]
            if jobs:
                t0 = time.time()
                texts = [JUDGE.format(trait=t, desc=prompts_all[t], text=r[:1500]) for k, t in jobs for r in replies[k]]
                scores = judge_batch(model, tok, texts)
                i = 0
                for k, t in jobs:
                    n = len(replies[k])
                    res.setdefault(k, {})[t] = scores[i : i + n]
                    i += n
                self.save(name, res)
                log(f"judge {fdir.name}: {len(texts)} ratings in {time.time() - t0:.0f}s")
            # coherence: the judge's 1-5 rating of each reply as fluent, coherent English, whatever its style
            cname = f"{fdir.name}/coherence.json"
            coh = {} if self.args.force else (self.load(cname) or {})
            todo = [k for k in replies if k not in coh]
            if todo:
                t0 = time.time()
                scores = judge_batch(model, tok, [COHERENCE.format(text=r[:1500]) for k in todo for r in replies[k]])
                i = 0
                for k in todo:
                    coh[k] = scores[i : i + len(replies[k])]
                    i += len(replies[k])
                self.save(cname, coh)
                log(f"coherence {fdir.name}: {len(todo)} conditions in {time.time() - t0:.0f}s")
            # fluency: perplexity of each condition's replies under the unsteered bf16 model
            fname = f"{fdir.name}/fluency.json"
            flu = {} if self.args.force else (self.load(fname) or {})
            todo = [k for k in replies if k not in flu]
            if todo:
                t0 = time.time()
                prompts = self.cfg["questions"][: self.cfg["n_prompts"]]
                for k in todo:
                    flu[k] = perplexity(model, tok, prompts[: len(replies[k])], replies[k])
                self.save(fname, flu)
                log(f"fluency {fdir.name}: {len(todo)} conditions in {time.time() - t0:.0f}s")


COHERENCE = (
    "Rate how fluent and coherent the following reply is as English text, whatever its tone or style.\n\n"
    "Reply:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Answer with a single digit from 1 (gibberish, broken, another language or endlessly repeating) "
    "to 5 (perfectly fluent and coherent)."
)


@torch.no_grad()
def judge_batch(model, tok, texts: list[str], batch_size: int = 8) -> list[float]:
    """Expected 1-5 rating from the model's probabilities of the digits, as
    :func:`evaluate.judge_scores`, batched with left padding."""
    digits = [tok.encode(d)[0] for d in "12345"]
    weights = torch.arange(1, 6, dtype=torch.float32)
    enc = [tok.encode(D.chat_text(tok, t)) for t in texts]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = [0.0] * len(enc)
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        T = max(len(enc[i]) for i in idx)
        ids = torch.tensor([[pad] * (T - len(enc[i])) + enc[i] for i in idx], device=model.device)
        mask = torch.tensor([[0] * (T - len(enc[i])) + [1] * len(enc[i]) for i in idx], device=model.device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        p = logits[:, digits].softmax(-1).cpu()
        for r, i in enumerate(idx):
            out[i] = float((p[r] * weights).sum())
    return out


STAGES = ("panel", "heldout", "score", "dose", "replies", "attribute", "judge", "report")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("stages", help=f"comma-separated: {', '.join(STAGES)}")
    p.add_argument("--format", default="bf16", help="bf16, nf4, fp4, int8, or fq<bits> (fake quantization)")
    p.add_argument("--run-dir", type=Path, default=Path("runs/robustness"))
    p.add_argument("--model", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true")
    g = p.add_argument_group("study settings (fixed by the first run in a --run-dir, kept in its config.json)")
    g.add_argument("--studies", type=lambda s: s.split(","), default=None, help="hedwig,new")
    g.add_argument("--study-runs", type=json.loads, default=None, help='JSON {"study": "search run dir"}')
    g.add_argument("--top", type=int, default=None)
    g.add_argument("--weak", type=lambda s: [int(x) for x in s.split(",") if x], default=None)
    g.add_argument("--n-random", type=int, default=None)
    g.add_argument("--n-questions", type=int, default=None)
    g.add_argument("--heldout-tokens", type=int, default=None)
    g.add_argument("--n-prompts", type=int, default=None)
    g.add_argument("--reply-tokens", type=int, default=None)
    g.add_argument("--max-traits", type=int, default=None, help="first N traits per study (smoke runs)")
    g.add_argument("--layers", type=lambda s: [int(x) for x in s.split(":")], default=None,
                   help="attribution layers START:STOP (default 4:28)")


def run(args: argparse.Namespace) -> None:
    stages = [s for s in args.stages.split(",") if s]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {STAGES}")
    study = Study(args.run_dir, args.format, args)
    for s in stages:
        if s == "report":
            from .robustness_report import write_report

            write_report(args.run_dir)
            log(f"report: {args.run_dir / 'REPORT.md'}")
        else:
            log(f"{s} ({study.fmt})")
            getattr(study, f"stage_{s}")()
