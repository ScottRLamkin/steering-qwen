#!/usr/bin/env bash
# Last in line: AtP's exact sweep again, with float32 logits and the exact effects saved (~2.2 h).
set -u
cd "$(dirname "$0")/../.."
while pgrep -f "scripts/run_methods.sh|runs/methods/extras.sh|runs/methods/followups.sh" >/dev/null; do sleep 30; done
L=runs/methods/run_methods.log
[ ! -f runs/methods/atp/eval-bf16-logits.json ] && cp runs/methods/atp/eval.json runs/methods/atp/eval-bf16-logits.json
echo "=== atp (fp32 readout) $(date '+%H:%M:%S')" >> $L
uv run neurons atp eval --run-dir runs/methods/atp --force >> runs/methods/atp.log 2>&1 && echo "    ok $(date '+%H:%M:%S')" >> $L || echo "    FAILED atp (see runs/methods/atp.log)" >> $L
uv run neurons report runs/methods --out runs/METHODS_REPORT.md >> $L 2>&1
echo "=== followups2 done $(date '+%H:%M:%S')" >> $L
