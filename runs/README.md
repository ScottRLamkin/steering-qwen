# Neuron-search runs

These are the results of reproducing HedwigAI's personality-neuron search in Qwen3.5-4B. `REPORT.md` summarizes and compares the runs. To reproduce them:

```bash
caffeinate -i scripts/run_reproduction.sh
```

| Directory | What |
|---|---|
| `torch-bf16/` | `scripts/find_neurons.py`, unquantized bf16 on MPS |
| `mlx-8bit/` | HedwigAI's own MLX scripts, unchanged (gist rev 59d237ec, mlx-lm 0.32.2), 8-bit MLX weights |
| `torch-bf16-mlx-answers/` | Our attribution, screen and verify in bf16, on the MLX run's answers. Compare it with `mlx-8bit/` to check our scoring code. |
| `torch-nf4/` | `scripts/find_neurons.py` with bitsandbytes 4-bit NF4 on MPS |
| `search/new-traits/` | `neurons search --trait-set new`: the same search on our ten traits, bf16 |
| `robustness/` | `neurons robustness`: single neurons from both searches, pushed in six model formats |

## Files in a `find_neurons.py` / `neurons search` run

| File | Contents |
|---|---|
| `config.json` | the traits (system prompts), discovery questions and held-out prompts used |
| `answers.json` | `{"plain": [20 answers], "<trait>": [20 answers], ...}`, the same shape as the gist's `answers.json` |
| `medians.json` | median residual norm leaving each layer, from the first 6 plain question+answer texts |
| `attrib.json` | `shortlists["<trait>:vs_plain" or ":vs_others"]`: the top 15 by \|score\|, with sign and rank. `watch`: score and global rank of the controls, L17N7426 and the blog's 25 neurons, for every shortlist, whatever their rank |
| `screen.json` | `rows`: every candidate with `sources` (the shortlists and ranks it entered from) and its log-prob shift on every answer set. `ranked`: each trait's full ranking by specificity |
| `verify.json` | held-out replies for each condition, with `chosen_for` (why it was verified, e.g. `poetic#1`) |
| `ratings.json` | the unsteered model's 1–5 coherence and trait ratings of every verified condition's replies; `broken` when coherence < 2.5 |
| `meta.json` | backend, device, precision, versions, model revision, git commit, and per-stage timings |
| `run.log` | the run's progress log |

## Files in the MLX run (the authors' names)

| File | Contents |
|---|---|
| `neurons.json` | firing-contrast ranking (the method the blog shows is a poor guide) |
| `neurons.json.answers.json` | the answers |
| `neurons.json.attrib.json` | attribution shortlists |
| `neurons.json.screen.json` | screen results |
| `verify.json` | held-out replies, for the conditions in `conds.json` |
| `env.txt`, `timings.txt`, `*.log` | environment, step timings and output |
