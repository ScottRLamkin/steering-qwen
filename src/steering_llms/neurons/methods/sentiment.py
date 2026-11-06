"""The sentiment unit (Radford et al., arXiv 1704.01444), looked for among Qwen3.5's neurons.

Find: mean ``down_in`` activation over each SST-2 sentence's tokens. Every neuron's AUROC for
positive vs negative; the most separating ``--screen`` neurons go into an L1-regularized logistic
regression (FISTA), whose largest weights are the candidate "sentiment units". Reported: held-out
accuracy of the single best unit with a midpoint threshold, of the sparse probe, and of all
screened neurons (the paper: one mLSTM unit reached ~92% on SST-2).

Change: clamp the unit(s) to a high or low value of their per-token distribution at every
position while the model writes a movie review; scored by an SST-2 sentiment classifier.
Retraining the paper's byte-level mLSTM is out of scope.
"""

import argparse

import torch

from .. import data
from ..collect import Collector, text_sample
from ..evaluate import Classifier, random_like
from ..gen import generate_batch
from ..intervene import Edit, Intervention
from ..neuronset import NeuronSet
from ..sites import Site
from ..stats import auroc, l1_logistic
from .common import Ctx

NAME = "sentiment"
HELP = "Sentiment unit: AUROC screen + L1 logistic probe on pooled activations; clamp"

TITLES = ["The Glass Orchard", "Midnight at Harrow Lane", "Copper Skies", "The Last Ferry", "Paper Lanterns",
          "Iron Meadow", "A Quiet Storm", "The Salt Road"]


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-texts", type=int, default=600, help="sentences per class")
    p.add_argument("--screen", type=int, default=4096)
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--k", type=int, default=16, help="probe units used for the multi-unit clamp")
    p.add_argument("--clamp-q", type=float, default=0.99)
    p.add_argument("--max-new-tokens", type=int, default=70)


def pooled(ctx: Ctx, rows: list[tuple[str, int]], batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    samples = [text_sample(ctx.tok, s, str(y)) for s, y in rows]
    st = Collector(ctx.layers, Site.DOWN_IN, None, pool="mean", moments=False).run(ctx.model, samples, batch)
    X = torch.cat([st["1"].sample_matrix(), st["0"].sample_matrix()])
    y = torch.cat([torch.ones(st["1"].n_samples), torch.zeros(st["0"].n_samples)])
    return X, y


def discover(ctx: Ctx, args: argparse.Namespace) -> dict[str, NeuronSet]:
    rows = data.sst2("train", seed=args.seed)
    pos = [r for r in rows if r[1] == 1][: args.n_texts]
    neg = [r for r in rows if r[1] == 0][: args.n_texts]
    X, y = pooled(ctx, pos + neg, args.batch_size * 2)
    W = ctx.width
    au = auroc(X[y == 1], X[y == 0])  # [L, W]
    sep = (au - 0.5).abs().flatten()
    scr = sep.topk(args.screen).indices
    Xs = X.reshape(X.shape[0], -1)[:, scr]
    w, b = l1_logistic(Xs, y, lam=args.lam, steps=600)
    order = w.abs().argsort(descending=True)
    nz = int((w.abs() > 0).sum())
    chosen = scr[order[: max(args.k, 1)]]
    # held-out accuracy
    val = data.sst2("validation")
    Xv, yv = pooled(ctx, val, args.batch_size * 2)
    Xv = Xv.reshape(Xv.shape[0], -1)
    best = int(sep.argmax())
    mu1, mu0 = X.reshape(X.shape[0], -1)[y == 1, best].mean(), X.reshape(X.shape[0], -1)[y == 0, best].mean()
    pred1 = (Xv[:, best] > (mu1 + mu0) / 2) if mu1 > mu0 else (Xv[:, best] < (mu1 + mu0) / 2)
    acc_unit = (pred1.float() == yv).float().mean().item()
    mean, std = Xs.mean(0), Xs.std(0).clamp_min(1e-6)
    pv = ((Xv[:, scr] - mean) / std) @ w + b
    acc_probe = ((pv > 0).float() == yv).float().mean().item()
    w_all, b_all = l1_logistic(Xs, y, lam=0.0, steps=300)
    acc_all = ((((Xv[:, scr] - mean) / std) @ w_all + b_all > 0).float() == yv).float().mean().item()
    # per-token clamp values for the chosen units
    samples = [text_sample(ctx.tok, s, "x") for s, _ in (pos[:100] + neg[:100])]
    layers_needed = sorted({int(c) // W for c in chosen})
    st = Collector(ctx.layers, Site.DOWN_IN, layers_needed, hist=(-8.0, 8.0, 1600), moments=False).run(ctx.model, samples, args.batch_size * 2)["x"]
    hi = st.percentile(args.clamp_q * 100)
    lo = st.percentile((1 - args.clamp_q) * 100)
    li = {L: j for j, L in enumerate(layers_needed)}
    ws = w[order[: len(chosen)]]
    ns = NeuronSet(
        Site.DOWN_IN, [int(c) // W for c in chosen], [int(c) % W for c in chosen], ws.tolist(),
        {
            "auroc": [au.flatten()[c].item() for c in chosen],
            "hi": [hi[li[int(c) // W], int(c) % W].item() for c in chosen],
            "lo": [lo[li[int(c) // W], int(c) % W].item() for c in chosen],
        },
        "sentiment",
        {"method": NAME, "acc_best_unit": acc_unit, "best_unit": [best // W, best % W], "best_auroc": au.flatten()[best].item(),
         "acc_probe": acc_probe, "probe_nonzero": nz, "acc_all_screened": acc_all, "n_val": len(val)},
    )
    ctx.log(f"best single unit L{best // W}N{best % W}: AUROC {au.flatten()[best]:.3f}, held-out acc {acc_unit:.3f}; "
            f"L1 probe ({nz} nonzero) {acc_probe:.3f}; all {args.screen} screened {acc_all:.3f}")
    return {"sentiment": ns}


def edits(sets: dict[str, NeuronSet], key: str = "sentiment", direction: str = "+", k: int = 1) -> list[Edit]:
    """Clamp the top ``k`` probe units: toward positive sentiment, units with positive weight go
    to their high value and units with negative weight to their low value (and vice versa)."""
    ns = sets[key]
    out = []
    for i in range(min(k, len(ns))):
        up = (ns.score[i] > 0) == (direction == "+")
        v = ns.extras["hi"][i] if up else ns.extras["lo"][i]
        out.append(Edit(ns.layer[i], [ns.index[i]], "set", v, Site.DOWN_IN, "all"))
    return out


def evaluate(ctx: Ctx, sets: dict[str, NeuronSet], args: argparse.Namespace) -> dict:
    prompts = [f"Write a two-sentence review of the movie \"{t}\"." for t in TITLES]
    ns = sets["sentiment"]
    rnd = {"sentiment": random_like(ns, ctx.width, args.seed)}
    conds = {"base": []}
    for k in (1, args.k):
        conds[f"+top{k}"] = edits(sets, direction="+", k=k)
        conds[f"-top{k}"] = edits(sets, direction="-", k=k)
    conds[f"random+top{args.k}"] = edits(rnd, direction="+", k=args.k)
    clf = Classifier("sentiment", ctx.model.device)
    out = {"probe": ns.meta, "conditions": {}}
    for name, e in conds.items():
        with Intervention(ctx.layers, e):
            texts = generate_batch(ctx.model, ctx.tok, prompts, max_new_tokens=args.max_new_tokens,
                                   repetition_penalty=1.1, batch_size=args.batch_size)
        r = {"positive": clf.mean(texts, "positive"), "examples": texts[:3]}
        out["conditions"][name] = r
        ctx.log(f"{name}: positive {r['positive']:.3f}")
    return out
