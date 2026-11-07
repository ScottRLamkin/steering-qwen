"""``neurons``: run a paper's neuron-finding recipe on Qwen3.5, then steer with what it found.

    uv run neurons list
    uv run neurons search all --run-dir runs/search/new-traits --trait-set new   # HedwigAI's search, our traits
    uv run neurons hedwig all --hedwig-run runs/search/new-traits --run-dir runs/methods/hedwig-new
    uv run neurons dpn-le all --run-dir runs/methods/dpn-le            # discover, then evaluate
    uv run neurons npti discover --traits Extraversion --n-questions 8
    uv run neurons chat --edits runs/methods/dpn-le/edits/Extraversion+.json "How was your weekend?"
    uv run neurons report runs/methods --out runs/METHODS_REPORT.md
    uv run neurons replay runs/methods/cps --run-dir runs/quant/nf4/cps --quant nf4   # same neurons, NF4 model

Stages write JSON into the run directory and are skipped when their output exists
(``--force`` reruns them).
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from ..model import QUANT_KINDS
from . import methods

STAGES = ("discover", "eval", "score", "all")


def common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help="model id or path (default Qwen/Qwen3.5-4B)")
    p.add_argument("--device", default=None)
    q = p.add_mutually_exclusive_group()
    q.add_argument("--quantize", dest="quantize", action="store_const", const="nf4", default=None, help="4-bit NF4")
    q.add_argument("--quant", dest="quantize", choices=QUANT_KINDS, help="bitsandbytes quantization: nf4, fp4 or int8")
    q.add_argument("--no-quantize", dest="quantize", action="store_false")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="neurons", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the recipes")

    from . import search

    s = sub.add_parser("search", help="HedwigAI's neuron search: generate, attribute, screen, verify",
                       description=(search.__doc__ or "").strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    search.add_args(s)

    from . import robustness

    rb = sub.add_parser("robustness", help="do single neurons' steering effects survive quantization?",
                        description=(robustness.__doc__ or "").strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    robustness.add_args(rb)

    for name in methods.MODULES:
        mod = methods.get(name)
        p = sub.add_parser(name, help=mod.HELP, description=(mod.__doc__ or "").strip(),
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument("stage", nargs="?", default="all", choices=STAGES)
        p.add_argument("--run-dir", type=Path, default=None, help="default runs/methods/<method>")
        common_args(p)
        p.add_argument("--traits", nargs="+", default=None, help="Big Five domains (personality recipes)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--eval-questions", type=int, default=6, help="test questions per trait and condition")
        p.add_argument("--no-replies", action="store_true", help="skip generation in eval (questionnaire only)")
        p.add_argument("--force", action="store_true", help="rerun stages whose output exists")
        p.add_argument("--quiet", action="store_true")
        mod.add_args(p)

    c = sub.add_parser("chat", help="generate with saved edits active")
    c.add_argument("prompt", nargs="*")
    c.add_argument("--edits", type=Path, action="append", default=[], help="edits JSON (repeatable)")
    c.add_argument("--max-new-tokens", type=int, default=150)
    c.add_argument("--baseline", action="store_true", help="also print the unedited reply")
    c.add_argument("-i", "--interactive", action="store_true")
    common_args(c)

    rp = sub.add_parser("replay", help="rerun a finished run with its saved settings, e.g. on a quantized model")
    rp.add_argument("src", type=Path, help="run directory holding config.json (and sets.json)")
    rp.add_argument("--run-dir", type=Path, required=True)
    rp.add_argument("--quant", choices=(*QUANT_KINDS, "none"), default=None, help="model format (default: as saved)")
    rp.add_argument("--rediscover", action="store_true", help="find the neurons again instead of reusing src's sets")
    rp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a saved setting (JSON value)")
    rp.add_argument("--stage", choices=STAGES, default="all")
    rp.add_argument("--force", action="store_true")

    qr = sub.add_parser("quant-report", help="compare bf16 runs with their quantized replays")
    qr.add_argument("--base", type=Path, default=Path("runs/methods"))
    qr.add_argument("--quant", type=Path, default=Path("runs/quant"))
    qr.add_argument("--out", type=Path, default=Path("runs/QUANT_REPORT.md"))

    r = sub.add_parser("report", help="write the comparison report from run directories")
    r.add_argument("root", type=Path, nargs="?", default=Path("runs/methods"))
    r.add_argument("--out", type=Path, default=Path("runs/METHODS_REPORT.md"))
    r.add_argument("--hedwig-run", type=Path, default=Path("runs/torch-bf16"))
    return ap


def load(args: argparse.Namespace):
    from .. import DEFAULT_MODEL, load_model

    return load_model(args.model or DEFAULT_MODEL, device=args.device, quantize=args.quantize)


def run_method(args: argparse.Namespace) -> None:
    from .. import discovery
    from ..neurons import data
    from ..neurons.neuronset import load_sets, save_sets
    from .methods.common import Ctx, model_info, set_summary

    mod = methods.get(args.cmd)
    if args.traits is None:
        args.traits = list(getattr(mod, "DEFAULT_TRAITS", data.BIG5))
    run_dir = args.run_dir or Path("runs/methods") / args.cmd
    tok, model = load(args)
    ctx = Ctx(model, tok, run_dir, quiet=args.quiet)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    ctx.save_json("config.json", {"args": cfg, **model_info(model)})
    sets_path = ctx.path("sets.json")
    if args.stage in ("discover", "all") and (args.force or not sets_path.exists()):
        ctx.log(f"{args.cmd}: discover")
        sets = mod.discover(ctx, args)
        save_sets(sets, sets_path)
        ctx.save_json("sets_summary.json", set_summary(sets))
    if args.stage in ("eval", "all"):
        eval_path = ctx.path("eval.json")
        if args.force or not eval_path.exists():
            sets = load_sets(sets_path) if sets_path.exists() else {}
            ctx.log(f"{args.cmd}: evaluate")
            res = mod.evaluate(ctx, sets, args)
            ctx.save_json("eval.json", res)
    if args.stage in ("score", "all") and hasattr(mod, "score"):
        score_path = ctx.path("score.json")
        if args.force or not score_path.exists():
            sets = load_sets(sets_path)
            ctx.log(f"{args.cmd}: persona log-prob score")
            ctx.save_json("score.json", mod.score(ctx, sets, args))
    discovery.log_stderr(f"done: {run_dir}")


def replay(a: argparse.Namespace) -> None:
    """Rebuild a run's arguments from its config.json and run the recipe again into ``a.run_dir``.

    Without ``--rediscover`` the source's neuron sets and other inputs (every top-level JSON except
    config, eval and score files, e.g. Knowledge Neurons' facts.json) are copied first, so only the
    evaluation runs: the same neurons and edits, measured on another model format. The persona
    reference answers in ``_shared/`` are always copied, so the log-prob score reads the same texts.
    """
    saved = json.loads((a.src / "config.json").read_text())["args"]
    name = saved["cmd"]
    defaults = vars(build_parser().parse_args([name]))
    args = {**defaults, **saved}
    for kv in a.set:
        k, _, v = kv.partition("=")
        k = k.replace("-", "_")
        if k not in args:
            raise SystemExit(f"--set {k}: {name} has no such setting")
        try:
            args[k] = json.loads(v)
        except json.JSONDecodeError:
            args[k] = v
    for k, v in defaults.items():
        if isinstance(v, Path) and isinstance(args[k], str):
            args[k] = Path(args[k])
    args.update(run_dir=a.run_dir, stage=a.stage, force=a.force, replayed_from=str(a.src), rediscovered=a.rediscover)
    if a.quant is not None:
        args["quantize"] = False if a.quant == "none" else a.quant
    a.run_dir.mkdir(parents=True, exist_ok=True)
    if not a.rediscover:
        for f in sorted(a.src.glob("*.json")):
            if f.name != "config.json" and not f.name.startswith(("eval", "score")) and not (a.run_dir / f.name).exists():
                shutil.copy2(f, a.run_dir / f.name)
    shared, dst = a.src.parent / "_shared", a.run_dir.parent / "_shared"
    if shared.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for f in shared.iterdir():
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
    run_method(argparse.Namespace(**args))


def chat(args: argparse.Namespace) -> None:
    from ..discovery import generate_answer
    from .intervene import Intervention, load_edits
    from ..model import find_layers

    tok, model = load(args)
    layers = find_layers(model)[0]
    edits = [e for p in args.edits for e in load_edits(p)]
    iv = Intervention(layers, edits)
    print(f"{len(edits)} edits on {sum(len(e.index) for e in edits)} neurons", file=sys.stderr)

    def one(prompt: str) -> None:
        if args.baseline:
            print("--- baseline\n" + generate_answer(model, tok, prompt, max_new_tokens=args.max_new_tokens))
        with iv:
            print("--- edited\n" + generate_answer(model, tok, prompt, max_new_tokens=args.max_new_tokens), flush=True)

    if args.prompt:
        one(" ".join(args.prompt))
    if args.interactive:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if line:
                one(line)


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_parser().parse_args(argv)
    if args.cmd == "list":
        for name in methods.MODULES:
            print(f"{name:12s} {methods.get(name).HELP}")
    elif args.cmd == "chat":
        chat(args)
    elif args.cmd == "search":
        from .search import run

        run(args)
    elif args.cmd == "robustness":
        from .robustness import run

        run(args)
    elif args.cmd == "report":
        from .report import write_report

        write_report(args.root, args.out, args.hedwig_run)
        print(json.dumps({"report": str(args.out)}))
    elif args.cmd == "replay":
        replay(args)
    elif args.cmd == "quant-report":
        from .quant_report import write_quant_report

        write_quant_report(args.base, args.quant, args.out)
        print(json.dumps({"report": str(args.out)}))
    else:
        run_method(args)


if __name__ == "__main__":
    main()
