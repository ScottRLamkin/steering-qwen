#!/usr/bin/env bash
# Run HedwigAI's own MLX scripts, unchanged, as the reference for scripts/find_neurons.py.
#
#   scripts/hedwig_mlx_reference.sh [OUT_DIR]        # default runs/mlx-8bit
#
# Apple Silicon only. Uses a throwaway uv environment with mlx-lm (nothing is added to the
# project), converts Qwen/Qwen3.5-4B to 8-bit MLX once (~4.5 GB, outside the repo), then runs
# their pipeline: extract -> attrib -> screen 25 -> steer. The steer conditions are their screen's
# top 3 per trait plus Lyra's two controls and L17N7426+, so outputs line up with our verify stage.
# Each step is skipped if its output exists.
set -euo pipefail

GIST_URL=https://gist.github.com/5e38fedf265feccd733061ecc6a6ad03.git
GIST_REV=59d237ecc6503d7766e1482c81fcad4a37e6f53b   # 2026-10-26, the version the blog links to
MLX_LM=mlx-lm==0.32.2                                # last release before the gist
OUT=${1:-runs/mlx-8bit}
MODEL=${MLX_MODEL:-$HOME/.cache/mlx-models/qwen35-4b-8bit}

mlx() { uv run --no-project --python 3.12 --with "$MLX_LM" "$@"; }
step() {
  local name=$1; shift
  echo "== $name ($(date +%H:%M:%S))"
  local t0=$SECONDS
  "$@"
  echo "$name $((SECONDS - t0))s" >> timings.txt
}

mkdir -p "$OUT"
if [ ! -d "$OUT/gist/.git" ]; then
  git clone -q "$GIST_URL" "$OUT/gist"
fi
git -C "$OUT/gist" checkout -q "$GIST_REV"

if [ ! -f "$MODEL/config.json" ]; then
  echo "== converting Qwen/Qwen3.5-4B to 8-bit MLX in $MODEL"
  mkdir -p "$(dirname "$MODEL")"
  mlx mlx_lm.convert --hf-path Qwen/Qwen3.5-4B -q --q-bits 8 --mlx-path "$MODEL"
fi

cd "$OUT"
{
  echo "mlx-lm: $MLX_LM"
  echo "gist: $GIST_REV"
  echo "model: $MODEL (8-bit MLX from Qwen/Qwen3.5-4B)"
  mlx python -c "import mlx.core as mx, mlx_lm, platform; print('mlx', mx.__version__, '| mlx_lm', mlx_lm.__version__, '|', platform.platform())"
} > env.txt

[ -f neurons.json.answers.json ] && [ -f neurons.json ] || step extract mlx python gist/find_neurons.py "$MODEL" neurons.json extract 2>&1 | tee extract.log
[ -f neurons.json.attrib.json ] || step attrib mlx python gist/attrib.py "$MODEL" neurons.json 2>&1 | tee attrib.log
[ -f neurons.json.screen.json ] || step screen mlx python gist/screen.py "$MODEL" neurons.json 25 2>&1 | tee screen.log

# Their screen's top 3 per trait, plus the controls and the target, as [name, [[L, n, dose]]].
python3 -I - <<'EOF'
import json
top = json.load(open("neurons.json.screen.json"))["top"]
name = lambda L, n, d: f"L{L}N{n}{'+' if d > 0 else '-'}"
conds = {"unsteered": []}
for L, n, d in [(14, 8949, 1.0), (13, 7956, -1.0), (17, 7426, 1.0)]:
    conds[name(L, n, d)] = [[L, n, d]]
for t, rows in top.items():
    for x in rows[:3]:
        conds.setdefault(name(x["layer"], x["neuron"], x["dose"]), [[x["layer"], x["neuron"], x["dose"]]])
json.dump([[k, v] for k, v in conds.items()], open("conds.json", "w"), indent=1)
print(len(conds), "steer conditions")
EOF

[ -f verify.json ] || step steer mlx python gist/find_neurons.py "$MODEL" verify.json steer conds.json neurons.json.answers.json 2>&1 | tee steer.log
echo "== done ($(date +%H:%M:%S))"
