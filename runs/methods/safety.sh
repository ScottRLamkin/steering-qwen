#!/usr/bin/env bash
# Safety Neurons, once there was disk for Qwen3.5-4B-Base: both models in bf16 (~15.6 GiB together).
set -u
cd "$(dirname "$0")/../.."
while pgrep -f "runs/methods/followups2.sh|hf download Qwen/Qwen3.5-4B-Base" >/dev/null; do sleep 15; done
L=runs/methods/run_methods.log
echo "=== safety (Base vs Instruct, bf16) $(date '+%H:%M:%S')" >> $L
HF_HUB_OFFLINE=1 uv run neurons safety all --run-dir runs/methods/safety --no-quantize >> runs/methods/safety.log 2>&1 && echo "    ok $(date '+%H:%M:%S')" >> $L || echo "    FAILED safety (see runs/methods/safety.log)" >> $L
HF_HUB_OFFLINE=1 uv run neurons report runs/methods --out runs/METHODS_REPORT.md >> $L 2>&1
echo "=== safety done $(date '+%H:%M:%S')" >> $L
