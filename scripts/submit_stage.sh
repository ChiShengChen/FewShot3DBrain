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
# Measured with scripts/probe_vram.py (job #26): peak reserved 4158 MB, hit by the
# batch-8 sliding-window evaluation on segmentation, not by training. +20% -> 5G.
# sequential_ft additionally carries gradients and Adam state for all 22.6M parameters
# (~270 MB more than LoRA's 47K), so it gets one extra GB.
VRAM="${VRAM:-5G}"
VRAM_FULL_FT="${VRAM_FULL_FT:-6G}"
# The DataLoader runs with num_workers=0, so there is no parallel loading to justify
# extra cores; these jobs are GPU-bound. The only sustained CPU work is volume loading
# and the single-threaded scipy EDT behind HD95. Reserving more than this just blocks
# other users - CPU, not VRAM, was the binding constraint on the first submission.
CPUS="${CPUS:-2}"
RAM="${RAM:-12}"
PRIORITY="${PRIORITY:-0}"   # default priority: no reason to jump ahead of other users
# Sliding-window overlap for test scoring. 0.25 is ~3x cheaper than nnU-Net's 0.5 default
# and still a standard setting. Every method in a comparison table must use the same
# value - use scripts/rescore.py if a run has already finished under a different one.
OVERLAP="${OVERLAP:-0.25}"
DRY="${DRY:-0}"
PY="${PY:-/home/meow/anaconda3/envs/pytorch291/bin/python}"
LEDGER="outputs/benchmark_v2/.submitted.tsv"
mkdir -p "$(dirname "$LEDGER")"; touch "$LEDGER"

submit_one() {
  local method="$1" seed="$2"
  if [[ -f "outputs/benchmark_v2/n${N_SHOT}/${method}/seed${seed}/results.json" ]]; then
    echo "[submit] skip ${method} seed${seed} (already has results.json)"
    return 0
  fi
  local name="fs3d-${method}-s${seed}"
  # A finished results.json is not the only reason to skip: the same configuration may
  # already be queued or running, and resubmitting would duplicate hours of training on
  # a contended machine. Match on the recorded job id rather than the job name, which
  # gq truncates to 18 characters -- long enough that e.g. sequential_linear at seeds 43
  # and 44 become indistinguishable.
  local prev
  prev=$(awk -F'\t' -v m="$method" -v s="$seed" -v n="$N_SHOT" \
           '$1==m && $2==s && $3==n {id=$4} END{print id}' "$LEDGER" 2>/dev/null)
  if [[ -n "${prev:-}" ]] && gq ls 2>/dev/null | awk -v id="#$prev" '$1==id{print $2}' \
       | grep -qE '^(running|queued)$'; then
    echo "[submit] skip ${method} seed${seed} (job #${prev} already queued or running)"
    return 0
  fi
  local vram="$VRAM"
  [[ "$method" == "sequential_ft" ]] && vram="$VRAM_FULL_FT"
  # Keep torch's thread pools inside the reservation rather than letting them size
  # themselves from the machine's 22 cores.
  local cmd=(gq submit -g "$vram" -c "$CPUS" -m "$RAM" -p "$PRIORITY" -n "$name" --
             env "OMP_NUM_THREADS=$CPUS" "MKL_NUM_THREADS=$CPUS"
             "$PY" scripts/run_benchmark_v2.py
             --method "$method" --tasks 2 3
             --n_shot "$N_SHOT" --seed "$seed" --epochs "$EPOCHS"
             --overlap "$OVERLAP")
  if [[ "$DRY" == "1" ]]; then
    printf '[dry] %s\n' "${cmd[*]}"
    return 0
  fi
  local out
  out=$("${cmd[@]}" 2>&1)
  echo "$out"
  local id
  id=$(printf '%s' "$out" | grep -oE 'submitted job #[0-9]+' | grep -oE '[0-9]+' | head -1)
  if [[ -n "$id" ]]; then
    printf '%s\t%s\t%s\t%s\n' "$method" "$seed" "$N_SHOT" "$id" >> "$LEDGER"
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
