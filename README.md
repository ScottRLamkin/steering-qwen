# Steering Qwen

Steer [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by adding one MLP neuron's output
direction (a column of `layers[L].mlp.down_proj`) to the residual stream with a forward hook.
For example, layer 14 neuron 8949 pushes toward "evil" and layer 17 neuron 7426 pushes toward "poetic".
See `Steering_LLMs.ipynb` for the experiments and references.

The model runs in-process with Hugging Face `transformers` + PyTorch, not in ollama. Steering
needs Python hooks on internal layers, which ollama/llama.cpp can't provide. The weights (~9 GB)
download into `~/.cache/huggingface` on first use.

## Setup

```bash
uv sync
```

| Device | Default | Notes |
|---|---|---|
| CUDA | 4-bit NF4 via bitsandbytes (2.85 GiB) | `--no-quantize` for bf16. `--max-gpu-memory 5GiB` caps GPU use, as the original notebook did. Uses the flash-linear-attention Triton kernels. |
| Apple Silicon (MPS) | bf16, unquantized (7.8 GiB) | `--quantize` gives the same 4-bit NF4 weights as CUDA. Triton doesn't exist on macOS, so transformers uses its slower pure-PyTorch linear-attention path. |
| CPU | bf16 | `--quantize` works here too. Slow. |

Doses in the notebook were tuned on the 4-bit model. bf16 steers the same way, with slightly different numbers.
With `--quantize`, MPS reproduces the CUDA run closely: same memory footprint, `median_norm(13)` 9.184 vs 9.189,
and generations that match word for word for the first 10–30 words before drifting.

## CLI

```bash
uv run steer "What do you think about humans?"                          # baseline
uv run steer --baseline --steer 14:8949:7.5 "What do you think about humans?"
uv run steer -s 14:8949:-3 -s 17:7426:6 "What do you think about pet cats?"
uv run steer --mode dose -s 14:8949:0.4 -s 13:7956:0.7 "What do you think of pet cats?"
uv run steer -i -s 17:7426:7.5                                          # interactive; :help for commands
uv run steer --quantize -s 14:8949:7.5                                  # 4-bit NF4, as on CUDA
```

`--steer LAYER:NEURON:X` can be repeated.

- With `--mode raw` (the default), `X` is the length of the push: `X * col / ||col||`.
- With `--mode dose`, `X` is a multiple of that layer's median residual norm. Dose 1.0 pushes as hard as a typical residual vector is long, so doses compare across layers.

Each run reloads the model, so use `-i` to keep it loaded between prompts.

## Scripts

```bash
uv run scripts/notebook_demos.py --list            # the notebook's raw-amount experiments
uv run scripts/notebook_demos.py --filter cats
uv run scripts/dose_sweep.py --layer 14 --neuron 8949 --doses -0.4 0.2 0.4
```

## Finding neurons

`scripts/find_neurons.py` searches the model for neurons that produce a trait. It is a PyTorch
port of [HedwigAI's search](https://www.hedwigai.com/blog/qwen-personality-neurons), which found
L17 N7426 (poetic) and 11 other one-neuron personalities. The code is in `src/steering_llms/discovery.py`.

1. **generate**: answers 20 questions plainly and under each trait's system prompt.
2. **attribute**: re-reads the answers without the system prompt. For every neuron in layers 4–27 it estimates, from one gradient, how much a dose +1 push would raise that trait's answers over the plain answers, and over the other traits' answers. It keeps the top 15 for each.
3. **screen**: actually steers every shortlisted neuron and ranks them by how much more likely they make their own trait's answers than the others'.
4. **verify**: writes held-out answers for the top 3 per trait, so you can read what each neuron does.

```bash
uv run scripts/find_neurons.py all --run-dir runs/torch-bf16                  # ~1 h on an M4 Pro
uv run scripts/find_neurons.py all --run-dir runs/mine --traits my_traits.json # your own traits
uv run scripts/compare_runs.py runs/torch-bf16 --out runs/REPORT.md
```

Each stage writes JSON into the run directory, and a stage whose output already exists is skipped. Every screened candidate records which shortlist it came from.

`--traits` takes a JSON object `{"name": "system prompt", ...}`. The defaults reproduce the authors' run exactly: 9 traits, greedy decoding, mlx-lm's repetition penalty, and the same dose calibration.

`scripts/run_reproduction.sh` runs the full reproduction: bf16, NF4, and the authors' own MLX scripts as a reference (`scripts/hedwig_mlx_reference.sh`, Apple Silicon only). It then writes `runs/REPORT.md`.

## Notebook

```bash
uv run jupyter lab Steering_LLMs.ipynb
```

In VS Code, pick the `.venv` interpreter as the kernel. The helpers live in `src/steering_llms/`.
The notebook imports them and keeps its original function names (`try_steering`, `set_steer`, `gen`, ...).

## Tests

```bash
uv run pytest
```

- **Spoofed CUDA:** `tests/test_load_model.py` fakes CUDA/MPS availability and checks the exact arguments `load_model` passes to transformers. On CUDA these must match the original notebook: NF4, bf16 compute, `device_map="auto"`, `max_memory`.
- **Tiny model on every device:** `tests/test_steer.py` builds a tiny random Qwen3.5 (same hybrid architecture, 16M params). It tests steering on every device found (cpu, mps, cuda), each in bf16 and in bitsandbytes 4-bit.
- **Real model:** `scripts/check_against_notebook.py` loads Qwen3.5-4B and checks memory, calibration numbers, steering and restoring the baseline. It also reports how closely generations match the CUDA outputs saved in the notebook. Pass `--quantize` or `--no-quantize` to pick the precision.
- **Real CUDA:** on any NVIDIA machine, run `uv run pytest` and the check script; the tests add `cuda` variants automatically. Without a local GPU, use Google Colab: [`colab/cuda_check.ipynb`](colab/cuda_check.ipynb) installs the locked environment on a Colab GPU, runs all of the above, and prints a pass/fail summary. Open it in Colab, set Runtime → T4 GPU, then Run all:
  https://colab.research.google.com/github/ramayer/steering-qwen/blob/main/colab/cuda_check.ipynb

The first run fetches the Qwen3.5 config and tokenizer (~12 MB). After that the suite takes about 10 s.

## Library

```python
from steering_llms import load_model, Steerer

tok, model = load_model()            # cuda > mps > cpu
st = Steerer(model, tok)
with st.steering([(14, 8949, 7.5), (17, 7426, 5)]):
    print(st.generate("What do you think about humans?", max_new_tokens=100))
```
