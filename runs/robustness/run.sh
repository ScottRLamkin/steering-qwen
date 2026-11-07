#!/usr/bin/env bash
# Our own neurons, and whether surviving quantization measures what a neuron means.
#   runs/search/new-traits     HedwigAI's search on our ten traits (bf16)
#   runs/methods/hedwig-new    the found neurons as neuron sets and edits, with their Big Five mapping
#   runs/robustness/<format>   the same single-neuron pushes measured in bf16, NF4, FP4, fake int4/int3, int8
# Resumable: finished stages are skipped. Run under caffeinate -i.
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
L=runs/robustness/run.log
mkdir -p runs/robustness
step() {  # name command...
  local name=$1; shift
  echo "=== $name $(date '+%H:%M:%S')" >> $L
  if "$@" >> "runs/robustness/$name.log" 2>&1; then echo "    ok $(date '+%H:%M:%S')" >> $L
  else echo "    FAILED $name (see runs/robustness/$name.log)" >> $L; fi
}
R="uv run neurons robustness"
step search uv run neurons search all --run-dir runs/search/new-traits --trait-set new --no-quantize
step hedwig-new uv run neurons hedwig all --hedwig-run runs/search/new-traits --run-dir runs/methods/hedwig-new
step bf16-setup $R panel,heldout --format bf16
step bf16 $R score,dose,replies,attribute --format bf16
step nf4 $R score,replies,attribute --format nf4
step fp4 $R score,replies,attribute --format fp4
step fq4 $R score --format fq4
step fq3 $R score,dose,replies,attribute --format fq3
step int8 $R score,replies,attribute --format int8
step judge $R judge --format bf16
step report $R report
echo "=== all done $(date '+%H:%M:%S')" >> $L
