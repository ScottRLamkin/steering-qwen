#!/usr/bin/env bash
# After run.sh: Safety Neurons' bf16 result again on 60 held-out prompts instead of 20, same neurons.
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
while pgrep -f "runs/quant/run.sh" >/dev/null; do sleep 30; done
L=runs/quant/run.log
d=runs/quant/bf16-n60/safety
mkdir -p "$(dirname $d)"
echo "=== bf16 transfer safety n_eval=60 $(date '+%H:%M:%S')" >> $L
uv run neurons replay runs/methods/safety --run-dir $d --quant none --set n_eval=60 > $d.log 2>&1 && echo "    ok $(date '+%H:%M:%S')" >> $L || echo "    FAILED bf16-n60 safety (see $d.log)" >> $L
uv run neurons quant-report >> $L 2>&1
echo "=== followup done $(date '+%H:%M:%S')" >> $L
