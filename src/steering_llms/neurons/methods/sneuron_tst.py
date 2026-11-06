"""sNeuron-TST, style-specific neurons for text style transfer (arXiv 2410.00593).

Find: raw texts of the source and target style (ParaDetox toxic vs neutral, or SST-2 negative
vs positive) are read token by token. A neuron's score for a style is how much more often its
``act = silu(gate)`` is > 0 on that style's tokens than on the other style's. The top ``frac`` of
all neurons per style are that style's neurons; neurons in both lists are removed (the paper's
"style-specific" step; with signed scores the lists rarely overlap).

Change: zero the source-style neurons at every position while the model rewrites a source-style
text, optionally decoding with DoLa contrasting the last layers (the paper's fix for the fluency
drop from deactivation).

A chat model may already rewrite perfectly without help (Qwen3.5-4B does for detox), leaving no
room to improve. The ``copy`` conditions ask it to repeat the source text word for word instead:
if the zeroed neurons carry the source style, the copies lose it while random neurons don't.

The paper's top-k is 10k neurons for 7-8B models; ``frac`` 0.02 scales that to Qwen3.5-4B
(~5900 of 294,912).
"""

import argparse

from ..collect import Collector, text_sample
from ..decoding import dola_generate
from ..evaluate import Classifier, distinct_n, random_like
from ..gen import generate_batch
from ..intervene import Edit, Intervention, edits_from_set
from ..neuronset import NeuronSet
from ..select import to_set, top_k_mask
from ..sites import Site
from ..stats import freq_diff
from . import style
from .common import Ctx

NAME = "sneuron-tst"
HELP = "sNeuron-TST: style-specific neurons by firing-rate contrast; deactivate (+ DoLa)"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", default="detox", choices=["detox", "sentiment"])
    p.add_argument("--n-texts", type=int, default=500, help="texts per style")
    p.add_argument("--frac", type=float, default=0.02, help="fraction of all neurons per style")
    p.add_argument("--n-eval", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--no-dola", action="store_true")
    p.add_argument("--no-copy", action="store_true", help="skip the repeat-verbatim probe")


def discover(ctx: Ctx, args: argparse.Namespace) -> dict[str, NeuronSet]:
    t = style.tasks(args.task, args.n_texts, args.seed)
    samples = [text_sample(ctx.tok, x, s) for s in t["styles"] for x in t["texts"][s]]
    layer_ids = list(range(ctx.n_layers))
    st = Collector(ctx.layers, Site.ACT, layer_ids).run(ctx.model, samples, args.batch_size * 2, log=ctx.log)
    k = int(args.frac * ctx.n_layers * ctx.width)
    a, b = t["styles"]
    sets = {}
    masks = {a: top_k_mask(freq_diff(st[a], st[b]), k, by_abs=False), b: top_k_mask(freq_diff(st[b], st[a]), k, by_abs=False)}
    shared = masks[a] & masks[b]
    for s, other in ((a, b), (b, a)):
        m = masks[s] & ~shared
        sets[s] = to_set(m, freq_diff(st[s], st[other]), layer_ids, Site.ACT,
                         extras={"rate": st[s].frac_pos}, label=s, sort_by_abs=False,
                         meta={"method": NAME, "task": args.task, "k": k, "removed_shared": int(shared.sum())})
        ctx.log(f"{s}: {len(sets[s])} neurons, layers {sets[s].layer_counts()}")
    return sets


COPY = "Repeat the following text exactly, word for word, without changing anything. Reply with the text only.\n\nText: {text}"


def edits(sets: dict[str, NeuronSet], source: str, direction: str = "-") -> list[Edit]:
    return edits_from_set(sets[source], "mul", 0.0, scope="all")


def evaluate(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    t = style.tasks(args.task, args.n_texts, args.seed)
    source = t["styles"][0]
    ev = t["eval"][: args.n_eval]
    prompts = [t["instruction"].format(text=x) for x, _ in ev]
    conds = {
        "base": [],
        "deactivate": edits(sets, source),
        "random": edits({source: random_like(sets[source], ctx.width, args.seed)}, source),
    }
    ctx.save_edits(f"edits/deactivate_{source}.json", conds["deactivate"])
    clf = Classifier(t["classifier"], ctx.model.device)
    out = {"task": args.task, "source": source, "n_neurons": len(sets[source]), "conditions": {}}
    copy_prompts = [COPY.format(text=x) for x, _ in ev]
    plans = [(c, False, False) for c in conds] + ([] if args.no_dola else [("base", True, False), ("deactivate", True, False)])
    plans += [] if args.no_copy else [(c, False, True) for c in conds]
    for cond, dola, copy in plans:
        name = ("copy: " if copy else "") + cond + ("+dola" if dola else "")
        ps = copy_prompts if copy else prompts
        with Intervention(ctx.layers, conds[cond]):
            if dola:
                texts = [dola_generate(ctx.model, ctx.tok, p, max_new_tokens=args.max_new_tokens) for p in ps]
            else:
                texts = generate_batch(ctx.model, ctx.tok, ps, max_new_tokens=args.max_new_tokens,
                                       repetition_penalty=1.0 if copy else 1.2, batch_size=args.batch_size)
        texts = [x.strip().split("\n")[0] for x in texts]
        r = {
            "target_style": clf.mean(texts, t["target_label"]),
            "content": sum(style.rouge1(h, src) for h, (src, _) in zip(texts, ev)) / len(ev),
            "fluency_ppl": style.text_ppl(ctx.model, ctx.tok, texts),
            "distinct2": distinct_n(texts, 2),
            "examples": [{"source": s, "output": h} for (s, _), h in zip(ev[:8], texts[:8])],
        }
        if ev[0][1] is not None:
            r["ref_overlap"] = sum(style.rouge1(h, ref) for h, (_, ref) in zip(texts, ev)) / len(ev)
        out["conditions"][name] = r
        ctx.log(f"{name}: {t['target_label']} {r['target_style']:.3f} content {r['content']:.3f} ppl {r['fluency_ppl']:.1f}")
    out["source_style_of_inputs"] = 1 - clf.mean([s for s, _ in ev], t["target_label"])
    return out
