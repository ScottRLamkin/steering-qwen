#!/usr/bin/env bash
# Run every neuron recipe on Qwen3.5-4B, one after another, then write runs/METHODS_REPORT.md.
#
#   scripts/run_methods.sh                 # all recipes, bf16 on MPS (~3 h on an M4 Pro)
#   scripts/run_methods.sh dpn-le pcn      # just these
#   EXTRA="--quantize" scripts/run_methods.sh
#
# Stages are resumable: a recipe whose sets.json / eval.json exist is skipped. A recipe that fails
# is logged and the script moves on. Classifiers and datasets are downloaded on first use; after
# that the run works with HF_HUB_OFFLINE=1.
set -u
cd "$(dirname "$0")/.."
ROOT=${ROOT:-runs/methods}
METHODS=("$@")
if [ ${#METHODS[@]} -eq 0 ]; then
  # safety needs Qwen3.5-4B-Base (~9 GB more on disk) and two models in memory; run it explicitly
  METHODS=(dpn-le pcn npti hedwig sneuron-tst aura gender cps emotion sentiment atp knowledge iris)
fi
mkdir -p "$ROOT"
for m in "${METHODS[@]}"; do
  echo "=== $m $(date '+%H:%M:%S')" | tee -a "$ROOT/run_methods.log"
  if uv run neurons "$m" all --run-dir "$ROOT/$m" ${EXTRA:-} >>"$ROOT/$m.log" 2>&1; then
    echo "    ok $(date '+%H:%M:%S')" | tee -a "$ROOT/run_methods.log"
  else
    echo "    FAILED (see $ROOT/$m.log)" | tee -a "$ROOT/run_methods.log"
  fi
done
uv run neurons report "$ROOT" --out runs/METHODS_REPORT.md
