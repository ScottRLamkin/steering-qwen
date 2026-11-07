"""Search Qwen3.5-4B for neurons that produce a trait: HedwigAI's pipeline in PyTorch.

    uv run neurons search all --run-dir runs/torch-bf16                     # bf16 (MPS default)
    uv run neurons search all --run-dir runs/torch-nf4 --quantize           # 4-bit NF4 (--quant fp4|int8 too)
    uv run neurons search all --run-dir runs/search/new-traits --trait-set new   # our ten traits
    uv run neurons search all --run-dir runs/x --answers other/answers.json      # reuse answers
    uv run neurons search all --run-dir runs/mine --traits my_traits.json        # your own traits

``scripts/find_neurons.py`` runs the same thing. Then turn the result into neuron sets and edit
files for ``neurons chat`` with ``neurons hedwig all --hedwig-run <run-dir>``.

Stages (each writes JSON into --run-dir and is skipped if its output exists; --force redoes it):

  generate   answers.json   plain answers + one set per trait (system prompt), 100 tokens, greedy
  attribute  attrib.json    median norms, gradient attribution, top-15 shortlist per trait x reference
             medians.json, grads.pt
  screen     screen.json    every pooled candidate steered at its signed unit dose; log-prob shifts,
                            and each trait's ranking of all candidates by specificity
  verify     verify.json    held-out replies: unsteered, the controls, L17N7426+, top 3 per trait
  all        all of the above, in order

--trait-set picks a built-in set (``hedwig``, the default, or ``new``; see ``neurons.traits``).
--traits takes a JSON object {name: system prompt}; --questions and --heldout take JSON lists.
Defaults are the authors' 9 traits, 20 questions and 8 held-out prompts.
"""

import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from .. import Steerer, load_model
from .. import discovery as D
from ..model import QUANT_KINDS, quant_kind
from .traits import TRAIT_SETS

STAGES = ["generate", "attribute", "screen", "verify"]
OUTPUT = {"generate": "answers.json", "attribute": "attrib.json", "screen": "screen.json", "verify": "verify.json"}


def model_revision(model_id: str) -> str | None:
    """The Hugging Face snapshot (commit) the cached weights came from."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id, local_files_only=True)).name
    except Exception:
        return None


class Run:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dir = Path(args.run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {"stages": {}}
        self.config = self._config()
        self._model = None

    # ---- bookkeeping ----

    def log(self, msg: str) -> None:
        line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        print(line, file=sys.stderr, flush=True)
        with open(self.dir / "run.log", "a") as f:
            f.write(line + "\n")

    def _config(self) -> dict:
        a = self.args
        want = {
            "traits": json.loads(Path(a.traits).read_text()) if a.traits else TRAIT_SETS[a.trait_set],
            "questions": json.loads(Path(a.questions).read_text()) if a.questions else D.QUESTIONS,
            "heldout": json.loads(Path(a.heldout).read_text()) if a.heldout else D.HELDOUT,
        }
        if a.only_traits:
            names = a.only_traits.split(",")
            want["traits"] = {k: want["traits"][k] for k in names}
        if a.n_questions:
            want["questions"] = want["questions"][: a.n_questions]
        if a.n_heldout:
            want["heldout"] = want["heldout"][: a.n_heldout]
        path = self.dir / "config.json"
        if path.exists():
            have = json.loads(path.read_text())
            custom = a.traits or a.trait_set != "hedwig" or a.questions or a.heldout
            if have != want and (custom or a.only_traits or a.n_questions or a.n_heldout):
                sys.exit(f"{path} has a different config; use a new --run-dir")
            return have
        path.write_text(json.dumps(want, indent=1, ensure_ascii=False))
        return want

    def save(self, name: str, obj) -> None:
        tmp = self.dir / (name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
        tmp.replace(self.dir / name)

    def load(self, name: str):
        return json.loads((self.dir / name).read_text())

    def done(self, stage: str) -> bool:
        return (self.dir / OUTPUT[stage]).exists() and not self.args.force

    def finish(self, stage: str, t0: float, **extra) -> None:
        self.meta["stages"][stage] = {
            "seconds": round(time.time() - t0, 1),
            "finished": dt.datetime.now().isoformat(timespec="seconds"),
            **extra,
        }
        self.meta["env"] = self.env()
        self.save("meta.json", self.meta)

    @property
    def traits(self) -> list[str]:
        return list(self.config["traits"])

    # ---- model ----

    @property
    def steerer(self) -> Steerer:
        if self._model is None:
            t0 = time.time()
            tok, model = load_model(self.args.model, device=self.args.device, quantize=self.args.quantize)
            self._model = Steerer(model, tok)
            self.log(f"loaded {self.args.model} on {model.device} in {time.time() - t0:.0f}s")
        return self._model

    def env(self) -> dict:
        e = {
            "backend": "pytorch/transformers",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        }
        import transformers

        e["transformers"] = transformers.__version__
        try:
            import bitsandbytes

            e["bitsandbytes"] = bitsandbytes.__version__
        except ImportError:
            pass
        try:
            import fla  # noqa: F401

            e["linear_attention"] = "flash-linear-attention (Triton)"
        except Exception:
            e["linear_attention"] = "transformers PyTorch fallback (flash-linear-attention not installed)"
        try:
            here = Path(__file__).resolve().parent
            git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=here).stdout.strip()  # noqa: E731
            e["git_commit"] = git("rev-parse", "--short", "HEAD")
            e["git_dirty"] = bool(git("status", "--porcelain", "--", ":/src", ":/scripts"))
        except Exception:
            pass
        if self._model is not None:
            m = self._model.model
            q = quant_kind(m)
            e.update(
                model=self.args.model,
                model_revision=model_revision(self.args.model),
                device=str(m.device),
                dtype=str(m.dtype).replace("torch.", ""),
                precision=f"{q} (bitsandbytes)" if q else str(m.dtype).replace("torch.", ""),
            )
            if m.device.type == "cuda":
                e["gpu"] = torch.cuda.get_device_name()
        return e

    # ---- stages ----

    def generate(self) -> None:
        if self.done("generate"):
            return self.log("generate: answers.json exists, skipping")
        t0 = time.time()
        if self.args.answers:
            answers = json.loads(Path(self.args.answers).read_text())
            missing = [k for k in [D.PLAIN, *self.traits] if k not in answers]
            if missing:
                sys.exit(f"{self.args.answers} lacks answer sets {missing}")
            answers = {k: answers[k][: len(self.config["questions"])] for k in [D.PLAIN, *self.traits]}
            self.save("answers.json", answers)
            self.log(f"generate: copied answers from {self.args.answers}")
            return self.finish("generate", t0, answers_from=str(self.args.answers))
        st = self.steerer
        self.log(f"generate: {len(self.traits) + 1} sets x {len(self.config['questions'])} questions")
        answers = D.generate_answers(
            st.model, st.tok, self.config["traits"], self.config["questions"],
            max_new_tokens=self.args.max_new_tokens, repetition_penalty=self.args.repetition_penalty, log=self.log,
        )
        self.save("answers.json", answers)
        self.finish("generate", t0, max_new_tokens=self.args.max_new_tokens, repetition_penalty=self.args.repetition_penalty)

    def calibrate(self) -> dict[int, float]:
        st = self.steerer
        if (self.dir / "medians.json").exists() and not self.args.force:
            med = {int(k): v for k, v in self.load("medians.json").items()}
            st.use_medians(med)
            return med
        answers = self.load("answers.json")
        texts = D.calibration_texts(st.tok, self.config["questions"], answers[D.PLAIN], D.CALIB_N)
        med = st.calibrate_on_texts(texts)
        self.save("medians.json", med)
        return med

    def watch_list(self) -> list[tuple[int, int]]:
        coords = [(L, n) for L, n, _ in [*D.CONTROLS, D.TARGET]] + [(b.layer, b.neuron) for b in D.BLOG_NEURONS]
        return list(dict.fromkeys(coords))

    def attribute(self) -> None:
        if self.done("attribute"):
            return self.log("attribute: attrib.json exists, skipping")
        t0 = time.time()
        st = self.steerer
        med = self.calibrate()
        self.log("attribute: medians " + " ".join(f"L{L}={med[L]:.2f}" for L in (13, 14, 17) if L in med))
        answers = self.load("answers.json")
        G = D.attribution_grads(st, self.config["questions"], answers, self.args.answer_tokens, log=self.log)
        torch.save(G, self.dir / "grads.pt")
        n_layers = len(st.layers)
        if self.args.layers:
            a, b = (int(x) for x in self.args.layers.split(":"))
            layers = [L for L in range(a, b) if L < n_layers]
        else:
            layers = [L for L in D.LAYERS if L < n_layers - 4]
        res = D.attribution_scores(st, G, self.traits, layers=layers, top=self.args.top, watch=self.watch_list())
        res = {"median_resid_norm": med, "answer_tokens": self.args.answer_tokens, "top": self.args.top, **res}
        self.save("attrib.json", res)
        for k, v in res["watch"].items():
            if k.startswith(("evil:", "poetic:")):
                show = {n: w["rank"] for n, w in v.items() if n in ("L14N8949", "L13N7956", "L17N7426")}
                self.log(f"attribute: {k} ranks {show} of {res['n_neurons']}")
        self.finish("attribute", t0)

    def screen(self) -> None:
        if self.done("screen"):
            return self.log("screen: screen.json exists, skipping")
        t0 = time.time()
        st = self.steerer
        self.calibrate()
        attrib = self.load("attrib.json")
        answers = self.load("answers.json")
        extra = [c for c in D.CONTROLS if c[0] < len(st.layers)]
        cands = D.pool_candidates(attrib["shortlists"], top=self.args.top, extra=extra)
        self.log(f"screen: {len(cands)} candidates")
        scorer = D.AnswerScorer(
            st.model, st.tok, self.config["questions"], answers,
            n_questions=self.args.screen_questions, answer_tokens=self.args.answer_tokens,
        )
        res = D.screen(st, scorer, cands, log=self.log)
        ranked = D.rank_candidates(res["rows"], self.traits, top=0)
        res.update(ranked=ranked, top={t: v[:8] for t, v in ranked.items()},
                   screen_questions=self.args.screen_questions, answer_tokens=self.args.answer_tokens)
        self.save("screen.json", res)
        for t, v in res["top"].items():
            self.log(f"screen: {t}: " + " ".join(f"{D.neuron_name(x['layer'], x['neuron'], x['dose'])}({x['specificity']:.2f})" for x in v[:4]))
        self.finish("screen", t0, n_candidates=len(cands))

    def verify(self) -> None:
        if self.done("verify"):
            return self.log("verify: verify.json exists, skipping")
        t0 = time.time()
        st = self.steerer
        med = self.calibrate()
        ranked = self.load("screen.json")["ranked"]
        conds: dict[str, dict] = {"unsteered": {"spec": [], "chosen_for": ["baseline"]}}
        fixed = [(*c, "control") for c in D.CONTROLS] + [(*D.TARGET, "target (checked, not searched)")]
        for L, n, d, why in fixed:
            if L < len(st.layers):
                conds[D.neuron_name(L, n, d)] = {"spec": [(L, n, d)], "chosen_for": [why]}
        for t in self.traits:
            for x in ranked[t][: self.args.verify_top]:
                name = D.neuron_name(x["layer"], x["neuron"], x["dose"])
                conds.setdefault(name, {"spec": [(x["layer"], x["neuron"], x["dose"])], "chosen_for": []})
                conds[name]["chosen_for"].append(f"{t}#{x['rank']}")
        self.log(f"verify: {len(conds)} conditions x {len(self.config['heldout'])} prompts")
        out = D.verify(st, self.config["heldout"], [(k, v["spec"]) for k, v in conds.items()],
                       max_new_tokens=self.args.verify_max_new_tokens, repetition_penalty=self.args.repetition_penalty,
                       log=self.log)
        for k in out:
            out[k]["chosen_for"] = conds[k]["chosen_for"]
        self.save("verify.json", {"median_resid_norm": med, "conditions": out})
        self.finish("verify", t0, n_conditions=len(conds), max_new_tokens=self.args.verify_max_new_tokens)


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("stage", choices=[*STAGES, "all"])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--device", choices=["cuda", "mps", "cpu"])
    q = p.add_mutually_exclusive_group()
    q.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=None, help="4-bit NF4 (default: CUDA only)")
    q.add_argument("--quant", dest="quantize", choices=QUANT_KINDS, help="bitsandbytes quantization: nf4, fp4 or int8")
    p.add_argument("--answers", help="reuse this answers.json (e.g. from the MLX reference run) instead of generating")
    p.add_argument("--trait-set", choices=sorted(TRAIT_SETS), default="hedwig", help="built-in traits (default: the authors' 9)")
    p.add_argument("--traits", help="JSON object {trait: system prompt} (overrides --trait-set)")
    p.add_argument("--questions", help="JSON list of discovery questions")
    p.add_argument("--heldout", help="JSON list of held-out prompts")
    p.add_argument("--only-traits", help="comma-separated subset of traits (smoke tests)")
    p.add_argument("--n-questions", type=int, help="use only the first N questions (smoke tests)")
    p.add_argument("--n-heldout", type=int, help="use only the first N held-out prompts (smoke tests)")
    p.add_argument("--max-new-tokens", type=int, default=100, help="discovery answer length (authors: 100)")
    p.add_argument("--verify-max-new-tokens", type=int, default=90, help="held-out answer length (authors: 90)")
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--answer-tokens", type=int, default=32, help="answer tokens scored (authors: 32)")
    p.add_argument("--layers", help="attribution layers START:STOP (default 4:28, as the authors)")
    p.add_argument("--top", type=int, default=15, help="shortlist size per trait x reference (authors: 15)")
    p.add_argument("--screen-questions", type=int, default=10, help="questions per set in the screen (authors: 10)")
    p.add_argument("--verify-top", type=int, default=3, help="candidates per trait verified (authors: 3)")
    p.add_argument("--force", action="store_true", help="redo stages whose output exists")


def run(args: argparse.Namespace) -> None:
    r = Run(args)
    r.log(f"$ {' '.join(sys.argv)}")
    for stage in STAGES if args.stage == "all" else [args.stage]:
        getattr(r, stage)()
    if args.stage == "all" or args.stage == "verify":
        r.log("done")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="find_neurons", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
