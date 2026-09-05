#!/usr/bin/env bash
# Staged re-run of the benchmark under the corrected protocol.
#
#   bash scripts/run_stage.sh A          # decisive: does the headline claim survive?
#   bash scripts/run_stage.sh B          # remaining seeds for the headline methods
#   bash scripts/run_stage.sh C          # ablations / oracles
#
# Stages are ordered so the question that decides the paper is answered first.
# Always launch through scripts/gpu_queue.sh so a running job is never pre-empted.

set -uo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-A}"
N_SHOT="${N_SHOT:-32}"
EPOCHS="${EPOCHS:-100}"
LOG_DIR="outputs/benchmark_v2/logs"
mkdir -p "$LOG_DIR"

run_one() {
  local method="$1" seed="$2"
  local log="$LOG_DIR/${method}_n${N_SHOT}_seed${seed}.log"
  if [[ -f "outputs/benchmark_v2/n${N_SHOT}/${method}/seed${seed}/results.json" ]]; then
    echo "[stage] skip ${method} seed${seed} (results.json exists)"
    return 0
  fi
  echo "[stage] $(date '+%F %T') START ${method} seed${seed} -> ${log}"
  python scripts/run_benchmark_v2.py \
    --method "$method" --tasks 2 3 \
    --n_shot "$N_SHOT" --seed "$seed" --epochs "$EPOCHS" \
    >"$log" 2>&1
  local rc=$?
  echo "[stage] $(date '+%F %T') END   ${method} seed${seed} rc=${rc}"
  return 0   # keep the queue moving even if one configuration fails
}

case "$STAGE" in
  A)
    for m in lora lora_head sequential_linear sequential_ft; do run_one "$m" 42; done ;;
  B)
    for s in 43 44; do
      for m in lora lora_head sequential_linear sequential_ft; do run_one "$m" "$s"; done
    done ;;
  C)
    for s in 42 43 44; do
      for m in lora_encoder_only pertask_lora; do run_one "$m" "$s"; done
    done ;;
  *)
    echo "unknown stage: $STAGE (expected A, B or C)" >&2; exit 2 ;;
esac

echo "[stage] $(date '+%F %T') stage ${STAGE} complete"
