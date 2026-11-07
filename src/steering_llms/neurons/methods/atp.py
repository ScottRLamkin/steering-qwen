"""Attribution Patching (AtP, Kramár et al. arXiv 2403.00745) for MLP neurons, checked against
exact activation patching.

For token-aligned clean / corrupted prompt pairs (one country swapped for another) and the metric
``logit(clean answer) - logit(corrupted answer)``, AtP estimates the effect of patching each
neuron's corrupted value into the clean run with one forward and one backward pass:
``(a_noise - a_clean) * dM/da``, summed over positions. Exact patching measures the real effect.

Reported per pair: Pearson and Spearman correlation between estimate and truth on a candidate
pool (AtP's top-k plus random neurons), and recall of the true top-k within AtP's top-k for k = 10,
30, 100. For the first ``--full-pairs`` pairs every neuron in every layer is patched exactly, so
recall is over all 294,912 neurons. The exact effects are read from float32 logits computed from
the final hidden state (bf16 logits move in steps of 1/16 at these values) and saved next to the
AtP scores as ``atp/<pair>.exact.pt``.

HedwigAI's gradient attribution is the same first-order estimate with the residual-stream push
``dose * down_proj column`` in place of a patch. AtP*'s QK fix is about attention softmax
saturation and doesn't apply to MLP neurons; GradDrop is not implemented.
"""

import argparse
import random

import torch

from ..attribution import atp, exact_patch, fp32_logit_readout, logit_diff_metric
from ..neuronset import NeuronSet
from ..sites import Site
from .common import Ctx

NAME = "atp"
HELP = "AtP: gradient x activation-difference attribution, verified by exact patching"

PAIRS = [
    ("France", "Paris", "Italy", "Rome"),
    ("Germany", "Berlin", "Spain", "Madrid"),
    ("Japan", "Tokyo", "Egypt", "Cairo"),
    ("Russia", "Moscow", "Greece", "Athens"),
    ("Canada", "Ottawa", "Norway", "Oslo"),
    ("Kenya", "Nairobi", "Peru", "Lima"),
]
TEMPLATE = "The capital of {} is the city of"


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-pairs", type=int, default=4)
    p.add_argument("--pool", type=int, default=300, help="AtP top-k and random neurons patched exactly")
    p.add_argument("--full-pairs", type=int, default=1, help="pairs with an exact sweep over all neurons")
    p.add_argument("--patch-batch", type=int, default=64)


def pairs(tok) -> list[dict]:
    out = []
    for c1, a1, c2, a2 in PAIRS:
        clean, noise = tok.encode(TEMPLATE.format(c1)), tok.encode(TEMPLATE.format(c2))
        t1, t2 = tok.encode(" " + a1)[0], tok.encode(" " + a2)[0]
        if len(clean) == len(noise) and t1 != t2:
            out.append({"name": f"{c1}->{c2}", "clean": clean, "noise": noise, "good": t1, "bad": t2})
    return out


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def discover(ctx: Ctx, args: argparse.Namespace) -> dict[str, NeuronSet]:
    layer_ids = list(range(ctx.n_layers))
    sets = {}
    for pr in pairs(ctx.tok)[: args.n_pairs]:
        clean, noise = torch.tensor([pr["clean"]]), torch.tensor([pr["noise"]])
        est, m = atp(ctx.model, ctx.layers, clean, noise, logit_diff_metric(pr["good"], pr["bad"]), Site.DOWN_IN, layer_ids)
        per = est.sum(1)  # [L, W]
        flat = per.flatten()
        top = flat.abs().topk(args.pool).indices
        sets[pr["name"]] = NeuronSet(
            Site.DOWN_IN, [int(i) // ctx.width for i in top], [int(i) % ctx.width for i in top], flat[top].tolist(),
            {}, pr["name"], {"method": NAME, "clean_metric": m},
        )
        torch.save(per, ctx.path(f"atp/{pr['name']}.pt"))
        ctx.log(f"{pr['name']}: clean metric {m:.2f}, top AtP {flat[top[:5]].tolist()}")
    return sets


def evaluate(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    W, L = ctx.width, ctx.n_layers
    out = {}
    for i, pr in enumerate([p for p in pairs(ctx.tok) if p["name"] in sets]):
        per = torch.load(ctx.path(f"atp/{pr['name']}.pt"))
        flat = per.flatten()
        clean, noise = torch.tensor([pr["clean"]]), torch.tensor([pr["noise"]])
        readout = fp32_logit_readout(ctx.model, [pr["good"], pr["bad"]])
        if readout is not None:
            def metric(z):
                return z[:, 0] - z[:, 1]
        else:
            metric = logit_diff_metric(pr["good"], pr["bad"])
        if i < args.full_pairs:
            cand = list(range(L * W))
        else:
            top = flat.abs().topk(args.pool).indices.tolist()
            cand = sorted(set(top) | set(rng.sample(range(L * W), args.pool)))
        eff, base = exact_patch(ctx.model, ctx.layers, clean, noise, metric, [(c // W, None, c % W) for c in cand],
                                Site.DOWN_IN, batch_size=args.patch_batch, log=ctx.log, readout=readout)
        torch.save({"cand": torch.tensor(cand), "exact": eff, "fp32_readout": readout is not None},
                   ctx.path(f"atp/{pr['name']}.exact.pt"))
        est = flat[cand]
        cand_t = torch.tensor(cand)

        def recall(k: int) -> float:
            k = min(k, len(cand))
            t = set(cand_t[eff.abs().topk(k).indices].tolist())
            return len(t & set(cand_t[est.abs().topk(k).indices].tolist())) / k

        k = 100
        true_top = set(cand_t[eff.abs().topk(min(k, len(cand))).indices].tolist())
        nz = eff.abs()[eff != 0]
        r = {
            "clean_metric": base,
            "n_patched": len(cand),
            "full_sweep": i < args.full_pairs,
            "fp32_readout": readout is not None,
            "resolution": nz.min().item() if len(nz) else None,
            "pearson": torch.corrcoef(torch.stack([est, eff]))[0, 1].item(),
            "spearman": spearman(est, eff),
            "recall@10": recall(10),
            "recall@30": recall(30),
            "recall@100": recall(100),
            "true_100th": eff.abs().topk(min(k, len(cand))).values[-1].item(),
            "sign_agree_top100": sum(
                bool((flat[c] > 0) == (eff[cand.index(c)] > 0)) for c in true_top
            ) / max(len(true_top), 1) if len(cand) < 20000 else None,
            "top_true": [{"layer": c // W, "neuron": c % W, "exact": eff[cand.index(c)].item(), "atp": flat[c].item()}
                         for c in sorted(true_top, key=lambda c: -abs(eff[cand.index(c)].item()))[:10]]
            if len(cand) < 20000 else None,
        }
        if r["top_true"] is None:  # full sweep: index directly
            order = eff.abs().topk(10).indices.tolist()
            r["top_true"] = [{"layer": c // W, "neuron": c % W, "exact": eff[c].item(), "atp": flat[c].item()} for c in order]
            r["sign_agree_top100"] = sum(bool((flat[c] > 0) == (eff[c] > 0)) for c in true_top) / len(true_top)
        out[pr["name"]] = r
        ctx.log(f"{pr['name']}: pearson {r['pearson']:.3f} spearman {r['spearman']:.3f} recall@10/30/100 "
                f"{r['recall@10']:.2f}/{r['recall@30']:.2f}/{r['recall@100']:.2f}, 100th true effect {r['true_100th']:.3f}, "
                f"resolution {r['resolution']} ({len(cand)} patched)")
    return out
