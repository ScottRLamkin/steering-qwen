"""Gender neurons for gendered generation (arXiv 2605.30717, code: Gender-Neuron-Intervention).

Find: GCGender sentences labelled male / female / neutral are read token by token at ``down_in``.
For each gender against the other two pooled, three per-neuron statistics are z-scored over all
neurons and combined, ``0.45 z(d) + 0.35 z(dlogit) + 0.20 z(r)``:

* ``d``       Cohen's d of activation
* ``dlogit``  log-odds difference of firing (activation > 0)
* ``r``       relative mean difference ``(mu_g - mu_rest) / (|mu_g| + |mu_rest|)``

Kept: firing rate ``p_g >= 0.08``, ``z(d) >= 0.5``, ``z(dlogit) >= 0.7``, and a composite at least
0.4 above the neuron's composite for any other gender; at most 2% of a layer's neurons and 1% of
all neurons per gender.

Change: to steer toward gender ``g``, zero the other genders' neurons and double ``g``'s
(the code's version; the paper describes only the zeroing). Evaluated by gendered-word ratios of
short stories about occupations.
"""

import argparse

import torch

from .. import data
from ..collect import Collector, merge_stats, text_sample
from ..evaluate import GENDER_WORDS, lexicon_ratio, random_like
from ..gen import generate_batch
from ..intervene import Edit, Intervention, edits_from_set
from ..neuronset import NeuronSet
from ..select import cap_per_layer, cap_total, to_set
from ..sites import Site
from ..stats import cohens_d, zscore
from .common import Ctx

NAME = "gender"
HELP = "Gender neurons: z-scored composite of d, firing log-odds and relative mean; zero others, x2 target"

OCCUPATIONS = ["nurse", "engineer", "CEO", "kindergarten teacher", "mechanic", "secretary", "pilot",
               "librarian", "firefighter", "ballet dancer", "surgeon", "plumber"]


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-per", type=int, default=400, help="sentences per gender label")
    p.add_argument("--margin", type=float, default=0.4)
    p.add_argument("--min-rate", type=float, default=0.08)
    p.add_argument("--max-new-tokens", type=int, default=80)


def discover(ctx: Ctx, args: argparse.Namespace) -> dict[str, NeuronSet]:
    groups = data.gcgender("train", n_per=args.n_per, seed=args.seed)
    samples = [text_sample(ctx.tok, s, g) for g, texts in groups.items() for s in texts]
    layer_ids = list(range(ctx.n_layers))
    st = Collector(ctx.layers, Site.DOWN_IN, layer_ids).run(ctx.model, samples, args.batch_size * 2, log=ctx.log)
    genders = sorted(st)
    comp, parts = {}, {}
    for g in genders:
        rest = merge_stats(*[st[h] for h in genders if h != g])
        p_g = st[g].frac_pos.clamp(1e-4, 1 - 1e-4)
        p_r = rest.frac_pos.clamp(1e-4, 1 - 1e-4)
        d = cohens_d(st[g], rest)
        dlogit = (p_g / (1 - p_g)).log() - (p_r / (1 - p_r)).log()
        mu_g, mu_r = st[g].mean, rest.mean
        r = (mu_g - mu_r) / (mu_g.abs() + mu_r.abs() + 1e-6)
        zd, zl, zr = zscore(d), zscore(dlogit), zscore(r)
        comp[g] = 0.45 * zd + 0.35 * zl + 0.20 * zr
        parts[g] = {"p": st[g].frac_pos, "zd": zd, "zl": zl, "d": d, "dlogit": dlogit}
    total_cap = int(0.01 * ctx.n_layers * ctx.width)
    layer_cap = int(0.02 * ctx.width)
    sets = {}
    for g in genders:
        others = torch.stack([comp[h] for h in genders if h != g]).amax(0)
        q = parts[g]
        mask = (q["p"] >= args.min_rate) & (q["zd"] >= 0.5) & (q["zl"] >= 0.7) & (comp[g] - others >= args.margin)
        mask = cap_total(cap_per_layer(mask, comp[g], layer_cap, by_abs=False), comp[g], total_cap, by_abs=False)
        sets[g] = to_set(mask, comp[g], layer_ids, Site.DOWN_IN, extras={"d": q["d"], "dlogit": q["dlogit"]},
                         label=g, sort_by_abs=False, meta={"method": NAME, "n_per": args.n_per, "margin": args.margin})
        ctx.log(f"{g}: {len(sets[g])} neurons, layers {sets[g].layer_counts()}")
    return sets


def edits(sets: dict[str, NeuronSet], target: str, direction: str = "+", boost: float = 2.0) -> list[Edit]:
    out = []
    for g, ns in sets.items():
        if g == target:
            if boost != 1.0:
                out += edits_from_set(ns, "mul", boost)
        else:
            out += edits_from_set(ns, "mul", 0.0)
    return out


def evaluate(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    prompts = [f"Write a short story about a {o}. Give the {o} a name." for o in OCCUPATIONS]
    rnd = {g: random_like(ns, ctx.width, args.seed) for g, ns in sets.items()}
    conds = {"base": []}
    for g in sets:
        conds[f"to_{g}"] = edits(sets, g)
    conds["random_to_neutral"] = edits(rnd, "neutral") if "neutral" in rnd else []
    out = {"n_neurons": {g: len(v) for g, v in sets.items()}, "conditions": {}}
    for name, e in conds.items():
        if e:
            ctx.save_edits(f"edits/{name}.json", e)
        with Intervention(ctx.layers, e):
            texts = generate_batch(ctx.model, ctx.tok, prompts, max_new_tokens=args.max_new_tokens,
                                   repetition_penalty=1.1, batch_size=args.batch_size)
        r = {"ratio": lexicon_ratio(texts, GENDER_WORDS), "examples": [{"prompt": p, "text": t} for p, t in zip(prompts[:4], texts[:4])]}
        out["conditions"][name] = r
        ctx.log(f"{name}: " + " ".join(f"{k} {v:.2f}" for k, v in r["ratio"].items()))
    return out
