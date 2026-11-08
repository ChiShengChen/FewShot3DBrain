#!/usr/bin/env bash
# Submit the "fit the head first" reframing experiments to the shared gq scheduler.
#
#   bash scripts/submit_reframe.sh SWEEP     # lr grid to every task, seeds 42-44 (adds 3e-3)
#   bash scripts/submit_reframe.sh PROJ      # LoRA enc+dec with the output projection frozen, 4 seg tasks, 5 seeds
#   bash scripts/submit_reframe.sh RIFT      # full fine-tuning from random init, ISLES / hippocampus / WMH, 5 seeds
#   bash scripts/submit_reframe.sh FORGET1   # BraTS -> IXI age, sequential FT at its own rates (2:1e-4 5:3e-4)
#   bash scripts/submit_reframe.sh FORGET2   # EWC lambda=100 from the FORGET1 BraTS checkpoints (run after FORGET1)
#   bash scripts/submit_reframe.sh SWEEP5    # seeds 45-46 for the (strategy, task, rate) cells a file lists
#   DRY=1 bash scripts/submit_reframe.sh SWEEP
#
# One gq job per run, as in submit_stage.sh. Resource declarations follow the measured
# values there (5G segmentation, +1G for full fine-tuning); whole-head 96^3 runs train on
# 3.4x the voxels of a 64^3 patch and are declared at 8G / 9G.

set -uo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-}"
DRY="${DRY:-0}"
PY="${PY:-/home/meow/anaconda3/envs/pytorch291/bin/python}"
PRIORITY="${PRIORITY:-0}"
CPUS=2; RAM=4
LEDGER="outputs/benchmark_v2/.submitted_reframe.tsv"
mkdir -p "$(dirname "$LEDGER")"; touch "$LEDGER"

SEG_TASKS="2 4 7 8"           # BraTS, ISLES, hippocampus, WMH (64^3 patches, overlap 0.25)
WH_TASKS="5 6 9 10 11"        # IXI age, IXI sex, OASIS age, OASIS sex, OASIS CDR (96^3 whole head)
is_wh() { case " $WH_TASKS " in *" $1 "*) return 0;; *) return 1;; esac; }

# Method -> driver flags and the run-directory prefix the driver derives from them.
method_flags() {
  case "$1" in
    lora_head) echo "--method lora_head --lora_r 16" ;;
    lora)      echo "--method lora --lora_r 16" ;;
    *)         echo "--method $1" ;;
  esac
}
run_prefix() {
  case "$1" in
    lora_head) echo "lora_head_r16" ;;
    lora)      echo "lora_r16" ;;
    *)         echo "$1" ;;
  esac
}
task_flags() {  # per-task patch / evaluation settings, identical to the existing runs of that task
  if is_wh "$1"; then echo "--patch_size 96 --eval_batch_size 2"; else echo "--patch_size 64 --overlap 0.25"; fi
}
vram_for() {    # method task
  local v=5; is_wh "$2" && v=8
  [[ "$1" == "sequential_ft" ]] && v=$((v + 1))
  echo "${v}G"
}
expect_for() {  # rough wall time from the medians of the finished runs
  if is_wh "$2"; then echo 12m; elif [[ "$1" == "sequential_ft" ]]; then echo 40m; else echo 30m; fi
}

# submit_one <run_dir_name> <job_name> <vram> <expect> -- <driver args...>
submit_one() {
  local dir="$1" name="$2" vram="$3" expect="$4"; shift 5
  if [[ -f "outputs/benchmark_v2/n32/${dir}/seed${SEED}/results.json" ]]; then
    echo "[skip] ${dir} seed${SEED}: results.json exists"; return 0
  fi
  local prev
  prev=$(awk -F'\t' -v d="$dir" -v s="$SEED" '$1==d && $2==s {id=$3} END{print id}' "$LEDGER")
  if [[ -n "${prev:-}" ]] && gq ls 2>/dev/null | awk -v id="#$prev" '$1==id{print $2}' | grep -qE '^(running|queued)$'; then
    echo "[skip] ${dir} seed${SEED}: job #${prev} queued/running"; return 0
  fi
  local cmd=(gq submit -g "$vram" -c "$CPUS" -m "$RAM" -p "$PRIORITY" -t "$expect" -n "$name" --
             env "OMP_NUM_THREADS=$CPUS" "MKL_NUM_THREADS=$CPUS"
             "$PY" scripts/run_benchmark_v2.py "$@" --n_shot 32 --epochs 100 --seed "$SEED")
  if [[ "$DRY" == "1" ]]; then printf '[dry] %s -> %s\n' "${cmd[*]:1}" "$dir"; return 0; fi
  local out id
  out=$("${cmd[@]}" 2>&1); echo "$out"
  id=$(printf '%s' "$out" | grep -oE 'job #[0-9]+' | grep -oE '[0-9]+' | head -1)
  [[ -n "$id" ]] && printf '%s\t%s\t%s\n' "$dir" "$SEED" "$id" >> "$LEDGER"
}

METHODS="lora_head sequential_linear sequential_ft bitfit adapter"

case "$STAGE" in
  SWEEP)
    # 1e-4 and 3e-4 on the six tasks the first sweep did not cover; 3e-3 on all nine.
    for SEED in 42 43 44; do
      for m in $METHODS; do
        for t in $SEG_TASKS $WH_TASKS; do
          for lr in 1e-4 3e-4 3e-3; do
            [[ "$lr" != "3e-3" && ( "$t" == "2" || "$t" == "5" || "$t" == "6" ) ]] && continue
            submit_one "$(run_prefix $m)_t${t}_lr${lr}" "r-${m:0:3}-t${t}-${lr}-s${SEED}" "$(vram_for $m $t)" "$(expect_for $m $t)" -- \
              $(method_flags $m) --tasks "$t" $(task_flags $t) --lr "$lr" --tag "lr${lr}"
          done
        done
      done
    done ;;
  PROJ)
    # LoRA on encoder + decoder with the output projection left frozen (plain `lora`), r=16,
    # so the contrast with lora_head_r16 isolates the 66-parameter projection on every task.
    for SEED in 42 43 44 45 46; do
      for t in $SEG_TASKS; do
        submit_one "lora_r16_t${t}" "r-proj-t${t}-s${SEED}" "$(vram_for lora $t)" "$(expect_for lora $t)" -- \
          $(method_flags lora) --tasks "$t" $(task_flags $t)
      done
    done ;;
  RIFT)
    for SEED in 42 43 44 45 46; do
      for t in 4 7 8; do
        submit_one "sequential_ft_t${t}_randinit" "r-rift-t${t}-s${SEED}" "$(vram_for sequential_ft $t)" "$(expect_for sequential_ft $t)" -- \
          --method sequential_ft --tasks "$t" $(task_flags $t) --pretrained_path '' --tag randinit
      done
    done ;;
  FORGET1)
    # Same sequence as sequential_ft_t25 but each task at the rate full fine-tuning chose on
    # validation (BraTS 1e-4, IXI whole-head age 3e-4); BraTS is retrained at 1e-4 here.
    for SEED in 42 43 44 45 46; do
      submit_one "sequential_ft_t25_ownlr" "r-fgt-ft-s${SEED}" 9G 50m -- \
        --method sequential_ft --tasks 2 5 --patch_sizes 2:64 5:96 --overlap 0.25 --lrs 2:1e-4 5:3e-4 --tag ownlr
    done ;;
  FORGET2)
    for SEED in 42 43 44 45 46; do
      [[ -f outputs/benchmark_v2/n32/sequential_ft_t25_ownlr/seed${SEED}/results.json ]] || { echo "[wait] FORGET1 seed${SEED} not done"; continue; }
      submit_one "ewc_t25_ownlr_ewc100" "r-fgt-ewc-s${SEED}" 9G 40m -- \
        --method ewc --tasks 2 5 --patch_sizes 2:64 5:96 --overlap 0.25 --lrs 2:1e-4 5:3e-4 \
        --init_first_from outputs/benchmark_v2/n32/sequential_ft_t25_ownlr --lambda_ewc 100 --tag ownlr_ewc100
    done ;;
  SWEEP5)
    # Cells whose chosen rate differs from 1e-3: one "<method> <task> <lr>" per line on stdin.
    while read -r m t lr; do
      [[ -z "$m" ]] && continue
      for SEED in 45 46; do
        submit_one "$(run_prefix $m)_t${t}_lr${lr}" "r-${m:0:3}-t${t}-${lr}-s${SEED}" "$(vram_for $m $t)" "$(expect_for $m $t)" -- \
          $(method_flags $m) --tasks "$t" $(task_flags $t) --lr "$lr" --tag "lr${lr}"
      done
    done ;;
  *) echo "usage: $0 SWEEP|PROJ|RIFT|FORGET1|FORGET2|SWEEP5" >&2; exit 2 ;;
esac
