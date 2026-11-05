#!/usr/bin/env bash
# The reproduction runs, in order, then the comparison report (~4-5 h on an M4 Pro, 24 GB):
#
#   caffeinate -i scripts/run_reproduction.sh
#
#   1. runs/torch-bf16               our pipeline, bf16
#   2. runs/mlx-8bit                 HedwigAI's MLX scripts, unchanged, 8-bit (the reference)
#   3. runs/torch-bf16-mlx-answers   our scoring (bf16) on the MLX run's answers
#   4. runs/torch-nf4                our pipeline, bitsandbytes 4-bit NF4
#   5. runs/REPORT.md
#
# Runs 2 and 3 separate the causes of any difference: 3 vs 2 is our scoring code and precision,
# 3 vs 1 is the effect of generating different answers. A failed run doesn't stop the others.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} PYTHONUNBUFFERED=1
mkdir -p runs
status=0

run() {
  local name=$1; shift
  echo "=== $name started $(date '+%F %T')"
  if "$@" >> "runs/$name.log" 2>&1; then
    echo "=== $name finished $(date '+%T')"
  else
    echo "=== $name FAILED $(date '+%T'), see runs/$name.log"
    status=1
  fi
}

run torch-bf16 uv run scripts/find_neurons.py all --run-dir runs/torch-bf16 --no-quantize
run mlx-8bit scripts/hedwig_mlx_reference.sh runs/mlx-8bit
if [ -f runs/mlx-8bit/neurons.json.answers.json ]; then
  run torch-bf16-mlx-answers uv run scripts/find_neurons.py all --run-dir runs/torch-bf16-mlx-answers \
    --no-quantize --answers runs/mlx-8bit/neurons.json.answers.json
fi
run torch-nf4 uv run scripts/find_neurons.py all --run-dir runs/torch-nf4 --quantize
uv run scripts/compare_runs.py runs/torch-bf16 runs/mlx-8bit runs/torch-bf16-mlx-answers runs/torch-nf4
echo "=== all done $(date '+%F %T'), status $status"
exit $status
