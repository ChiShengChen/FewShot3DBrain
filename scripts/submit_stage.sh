#!/usr/bin/env bash
# Submit benchmark re-runs to the shared gq scheduler.
#
#   bash scripts/submit_stage.sh A            # queue stage A
#   VRAM=6G bash scripts/submit_stage.sh A    # override the declared VRAM
#   DRY=1 bash scripts/submit_stage.sh A      # print what would be submitted
#
# One gq job per (method, seed) rather than one job for the whole stage: this machine
# is shared by many sessions, so a 7-hour monolithic reservation would block everyone
# else far longer than the individual runs do.
#
# VRAM is declared from the measurement in scripts/probe_vram.py, not guessed. Re-run
# that probe if the model, patch size or eval batch size changes.

set -uo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-A}"
N_SHOT="${N_SHOT:-32}"
EPOCHS="${EPOCHS:-100}"
VRAM="${VRAM:-8G}"          # measured peak + headroom; see PROTOCOL_AUDIT.md
CPUS="${CPUS:-4}"
RAM="${RAM:-16}"
PRIORITY="${PRIORITY:-0}"   # default priority: no reason to jump ahead of other users
DRY="${DRY:-0}"
PY="${PY:-/home/meow/anaconda3/envs/pytorch291/bin/python}"

submit_one() {
  local method="$1" seed="$2"
  if [[ -f "outputs/benchmark_v2/n${N_SHOT}/${method}/seed${seed}/results.json" ]]; then
    echo "[submit] skip ${method} seed${seed} (already has results.json)"
    return 0
  fi
  local name="fs3d-${method}-s${seed}"
  local cmd=(gq submit -g "$VRAM" -c "$CPUS" -m "$RAM" -p "$PRIORITY" -n "$name" --
             "$PY" scripts/run_benchmark_v2.py
             --method "$method" --tasks 2 3
             --n_shot "$N_SHOT" --seed "$seed" --epochs "$EPOCHS")
  if [[ "$DRY" == "1" ]]; then
    printf '[dry] %s\n' "${cmd[*]}"
  else
    "${cmd[@]}"
  fi
}

case "$STAGE" in
  A) for m in lora lora_head sequential_linear sequential_ft; do submit_one "$m" 42; done ;;
  B) for s in 43 44; do
       for m in lora lora_head sequential_linear sequential_ft; do submit_one "$m" "$s"; done
     done ;;
  C) for s in 42 43 44; do
       for m in lora_encoder_only pertask_lora; do submit_one "$m" "$s"; done
     done ;;
  *) echo "unknown stage: $STAGE (expected A, B or C)" >&2; exit 2 ;;
esac

echo
echo "[submit] queued. watch with:  gq status   ·   gq log -f <id>"
