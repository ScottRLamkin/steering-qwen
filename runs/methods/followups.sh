#!/usr/bin/env bash
# Reruns after the batch and extras, for evaluations fixed while the batch ran:
# AurA on RealToxicityPrompts, gender with per-target random controls, emotion's LAPE order,
# sentiment as raw continuations, sNeuron-TST with the copy probe.
set -u
cd "$(dirname "$0")/../.."
while pgrep -f "scripts/run_methods.sh|runs/methods/extras.sh" >/dev/null; do sleep 30; done
L=runs/methods/run_methods.log
echo "=== followups $(date '+%H:%M:%S')" >> $L
[ -f runs/methods/aura/eval.json ] && [ ! -f runs/methods/aura/eval-paradetox-prefixes.json ] && cp runs/methods/aura/eval.json runs/methods/aura/eval-paradetox-prefixes.json
[ -f runs/methods/gender/eval.json ] && [ ! -f runs/methods/gender/eval-v1.json ] && cp runs/methods/gender/eval.json runs/methods/gender/eval-v1.json
[ -f runs/methods/sentiment/eval.json ] && [ ! -f runs/methods/sentiment/eval-chat.json ] && cp runs/methods/sentiment/eval.json runs/methods/sentiment/eval-chat.json
run() {  # name, then the neurons arguments
    local n=$1; shift
    echo "=== $n $(date '+%H:%M:%S')" >> $L
    uv run neurons "$@" >> runs/methods/$n.log 2>&1 && echo "    ok $(date '+%H:%M:%S')" >> $L || echo "    FAILED $n (see runs/methods/$n.log)" >> $L
}
run emotion emotion all --run-dir runs/methods/emotion --force
run aura aura eval --run-dir runs/methods/aura --force
run gender gender eval --run-dir runs/methods/gender --force
run sentiment sentiment eval --run-dir runs/methods/sentiment --force
run sneuron-tst sneuron-tst eval --run-dir runs/methods/sneuron-tst --force
uv run neurons report runs/methods --out runs/METHODS_REPORT.md >> $L 2>&1
echo "=== followups done $(date '+%H:%M:%S')" >> $L
