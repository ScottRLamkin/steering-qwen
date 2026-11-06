"""``neurons``: run a paper's neuron-finding recipe on Qwen3.5, then steer with what it found.

    uv run neurons list
    uv run neurons dpn-le all --run-dir runs/methods/dpn-le            # discover, then evaluate
    uv run neurons npti discover --traits Extraversion --n-questions 8
    uv run neurons chat --edits runs/methods/dpn-le/edits/Extraversion+.json "How was your weekend?"
    uv run neurons report runs/methods --out runs/METHODS_REPORT.md

Stages write JSON into the run directory and are skipped when their output exists
(``--force`` reruns them).
"""

import argparse
import json
import os
import sys
from pathlib import Path

from . import methods

STAGES = ("discover", "eval", "all")


def common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help="model id or path (default Qwen/Qwen3.5-4B)")
    p.add_argument("--device", default=None)
    q = p.add_mutually_exclusive_group()
    q.add_argument("--quantize", dest="quantize", action="store_true", default=None, help="4-bit NF4")
    q.add_argument("--no-quantize", dest="quantize", action="store_false")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="neurons", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the recipes")

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
    discovery.log_stderr(f"done: {run_dir}")


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
    elif args.cmd == "report":
        from .report import write_report

        write_report(args.root, args.out, args.hedwig_run)
        print(json.dumps({"report": str(args.out)}))
    else:
        run_method(args)


if __name__ == "__main__":
    main()
