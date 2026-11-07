#!/usr/bin/env bash
# Are the bf16 neurons as responsive in quantized Qwen3.5-4B? Replays runs/methods/* on NF4, int8 and FP4.
#   runs/quant/<fmt>/<recipe>               the bf16 run's neurons and edits, evaluated on <fmt> ("transfer")
#   runs/quant/<fmt>-rediscovered/<recipe>  neurons found again on <fmt>, then evaluated
# Only the recipes that steered in bf16; AtP is always rediscovered (its estimates come from the model),
# with exact patching on pools only (no 294,912-neuron sweep). Resumable: finished stages are skipped.
set -u
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
L=runs/quant/run.log
TRANSFER="cps knowledge aura sentiment hedwig dpn-le npti safety"
REDISCOVER="cps knowledge aura sentiment dpn-le npti safety"

replay() {  # fmt mode recipe
  local fmt=$1 mode=$2 r=$3 dir extra=()
  if [ "$mode" = transfer ]; then dir=runs/quant/$fmt/$r; else dir=runs/quant/$fmt-rediscovered/$r; extra=(--rediscover); fi
  [ "$r" = atp ] && extra+=(--set full_pairs=0)
  [ -f "$dir/eval.json" ] && { [ "$r" != dpn-le ] && [ "$r" != npti ] && [ "$r" != hedwig ] || [ -f "$dir/score.json" ]; } && return 0
  echo "=== $fmt $mode $r $(date '+%H:%M:%S')" >> $L
  mkdir -p "$(dirname "$dir")"
  local q=$fmt; [ "$fmt" = bf16 ] && q=none
  if uv run neurons replay runs/methods/$r --run-dir "$dir" --quant "$q" ${extra[@]+"${extra[@]}"} > "$dir.log" 2>&1; then
    echo "    ok $(date '+%H:%M:%S')" >> $L
  else
    echo "    FAILED $fmt $mode $r (see $dir.log)" >> $L
  fi
}

replay bf16 rediscover atp   # baseline: AtP on pools in bf16, as the quantized runs do it
for fmt in nf4 int8; do
  replay $fmt rediscover atp
  for r in $TRANSFER; do replay $fmt transfer $r; done
done
for r in $REDISCOVER; do replay nf4 rediscover $r; done
replay fp4 rediscover atp
for r in $TRANSFER; do replay fp4 transfer $r; done
echo "=== all done $(date '+%H:%M:%S')" >> $L
