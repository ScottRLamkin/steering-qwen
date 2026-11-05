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

| Device | How the model loads |
|---|---|
| CUDA | 4-bit NF4 via bitsandbytes (`--no-quantize` for bf16), with flash-linear-attention kernels |
| Apple Silicon (MPS) | bf16, unquantized (~9 GB; 16 GB+ unified memory recommended). bitsandbytes and flash-linear-attention aren't installed on macOS. transformers uses its pure-PyTorch fallback. |
| CPU | bf16, slow |

Doses in the notebook were tuned on the 4-bit model. On MPS they steer the same way, but the
text won't match the saved outputs word for word.

## CLI

```bash
uv run steer "What do you think about humans?"                          # baseline
uv run steer --baseline --steer 14:8949:7.5 "What do you think about humans?"
uv run steer -s 14:8949:-3 -s 17:7426:6 "What do you think about pet cats?"
uv run steer --mode dose -s 14:8949:0.4 -s 13:7956:0.7 "What do you think of pet cats?"
uv run steer -i -s 17:7426:7.5                                          # interactive; :help for commands
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

## Notebook

```bash
uv run jupyter lab Steering_LLMs.ipynb
```

In VS Code, pick the `.venv` interpreter as the kernel. The helpers live in `src/steering_llms/`.
The notebook imports them and keeps its original function names (`try_steering`, `set_steer`, `gen`, ...).

## Library

```python
from steering_llms import load_model, Steerer

tok, model = load_model()            # cuda > mps > cpu
st = Steerer(model, tok)
with st.steering([(14, 8949, 7.5), (17, 7426, 5)]):
    print(st.generate("What do you think about humans?", max_new_tokens=100))
```
