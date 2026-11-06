#!/usr/bin/env bash
# Follow-ups queued after the main batch: DPN-LE's log-prob score (added mid-run) and a bigger PCN.
set -u
cd "$(dirname "$0")/../.."
while pgrep -f "scripts/run_methods.sh" >/dev/null; do sleep 30; done
echo "=== extras $(date '+%H:%M:%S')" >> runs/methods/run_methods.log
uv run neurons dpn-le score --run-dir runs/methods/dpn-le >> runs/methods/dpn-le.log 2>&1 && echo "    ok dpn-le score $(date '+%H:%M:%S')" >> runs/methods/run_methods.log || echo "    FAILED dpn-le score" >> runs/methods/run_methods.log
uv run neurons pcn all --run-dir runs/methods/pcn-k500-all --k 500 --pcn-scope all >> runs/methods/pcn-k500-all.log 2>&1 && echo "    ok pcn-k500-all $(date '+%H:%M:%S')" >> runs/methods/run_methods.log || echo "    FAILED pcn-k500-all" >> runs/methods/run_methods.log
uv run neurons report runs/methods --out runs/METHODS_REPORT.md >> runs/methods/run_methods.log 2>&1
echo "=== extras done $(date '+%H:%M:%S')" >> runs/methods/run_methods.log
