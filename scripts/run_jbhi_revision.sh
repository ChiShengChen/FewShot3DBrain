#!/usr/bin/env bash
# ============================================================================
# IEEE JBHI Revision — All Supplementary Experiments
# Usage:
#   bash scripts/run_jbhi_revision.sh [--batch N] [--dry-run]
#
#   --batch N    Run only batch N (1-5). Default: run all.
#   --dry-run    Print commands without executing.
#
# Estimated time (RTX 3090):
#   Batch 1 (LoRA n64, 5 seeds):          ~6-8 hrs
#   Batch 2 (Rank ablation r=4/8/16):     ~8-12 hrs
#   Batch 3 (Per-task LoRA upper bound):   ~3-4 hrs
#   Batch 4 (All methods seed 45,46):      ~12-16 hrs
#   Batch 5 (Shot16 ablation seed 45,46):  ~2-3 hrs
#   Total:                                 ~31-43 hrs
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)

# --- Args ---
BATCH="all"
DRY_RUN=0
for arg in "$@"; do
  case $arg in
    --batch) shift; BATCH="$1"; shift ;;
    --batch=*) BATCH="${arg#*=}" ;;
    --dry-run) DRY_RUN=1 ;;
  esac
done

# --- Paths ---
PRET="./weights/fomo25_mmunetvae_pretrained.ckpt"
DATA="./data/preprocessed"
EPOCHS=100
PY=python

# --- Logging ---
LOG_DIR="./outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/jbhi_revision_${TIMESTAMP}.log"

log() {
  echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

# --- Run helper: skip if metrics.json already exists ---
run_exp() {
  local desc="$1"
  local metrics_path="$2"
  shift 2
  local cmd=("$@")

  if [ -f "$metrics_path" ]; then
    log "SKIP $desc — $metrics_path exists"
    return 0
  fi

  log "START $desc"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  [dry-run] ${cmd[*]}"
    return 0
  fi

  local t_start=$SECONDS
  if "${cmd[@]}" >> "$MAIN_LOG" 2>&1; then
    local elapsed=$(( SECONDS - t_start ))
    log "DONE  $desc (${elapsed}s)"
  else
    local elapsed=$(( SECONDS - t_start ))
    log "FAIL  $desc (${elapsed}s) — check $MAIN_LOG"
  fi
}

# --- Metrics path helper ---
# LoRA:              <save_dir>/metrics.json
# sequential_linear: <save_dir>/sequential_linear/metrics.json
# sequential_ft:     <save_dir>/sequential_ft/metrics.json
# ewc/lwf/replay:    <save_dir>/metrics.json
metrics_for() {
  local save_dir="$1"
  local method="$2"
  case "$method" in
    sequential_linear) echo "$save_dir/sequential_linear/metrics.json" ;;
    sequential_ft)     echo "$save_dir/sequential_ft/metrics.json" ;;
    *)                 echo "$save_dir/metrics.json" ;;
  esac
}

# ============================================================================
# BATCH 1: LoRA n_shot=64, 5 seeds
#   Reviewer: "補 LoRA n_shot=64 實驗"
# ============================================================================
run_batch_1() {
  log "===== BATCH 1: LoRA n_shot=64 (5 seeds) ====="
  for SEED in 42 43 44 45 46; do
    local sd="./outputs/miccai_experiments/n64/lora/seed${SEED}"
    run_exp "LoRA n64 seed=$SEED" \
      "$(metrics_for "$sd" lora)" \
      $PY run_continual_lora.py --tasks 2 3 --n_shot 64 --lora_decoder \
        --pretrained_path "$PRET" --data_dir "$DATA" --epochs $EPOCHS \
        --seed $SEED --save_dir "$sd"
  done
}

# ============================================================================
# BATCH 2: LoRA rank ablation r=4/8/16, 5 seeds each
#   Reviewer: "補 LoRA rank ablation (r=4/8/16) 含多 seed"
# ============================================================================
run_batch_2() {
  log "===== BATCH 2: LoRA Rank Ablation (5 seeds) ====="

  for RANK in 4 8 16; do
    for SEED in 42 43 44 45 46; do
      local suffix=""
      [ "$RANK" -ne 8 ] && suffix="_r${RANK}"
      local sd="./outputs/ablations/lora/seed${SEED}_shot32${suffix}"
      run_exp "LoRA r=$RANK n32 seed=$SEED" \
        "$(metrics_for "$sd" lora)" \
        $PY run_continual_lora.py --tasks 2 3 --n_shot 32 --lora_decoder \
          --lora_r $RANK \
          --pretrained_path "$PRET" --data_dir "$DATA" --epochs $EPOCHS \
          --seed $SEED --save_dir "$sd"
    done
  done
}

# ============================================================================
# BATCH 3: Per-task LoRA (independent, non-continual) — Upper Bound
#   Reviewer: "補 per-task LoRA（非 continual）作為 upper bound"
# ============================================================================
run_batch_3() {
  log "===== BATCH 3: Per-task LoRA Upper Bound (5 seeds) ====="

  # Task 2 (segmentation) — independent
  for SEED in 42 43 44 45 46; do
    local sd="./outputs/pertask_lora/task2/seed${SEED}"
    run_exp "Per-task LoRA T2 seed=$SEED" \
      "$(metrics_for "$sd" lora)" \
      $PY run_continual_lora.py --tasks 2 --n_shot 32 --lora_decoder \
        --pretrained_path "$PRET" --data_dir "$DATA" --epochs $EPOCHS \
        --seed $SEED --save_dir "$sd"
  done

  # Task 3 (regression) — independent
  for SEED in 42 43 44 45 46; do
    local sd="./outputs/pertask_lora/task3/seed${SEED}"
    run_exp "Per-task LoRA T3 seed=$SEED" \
      "$(metrics_for "$sd" lora)" \
      $PY run_continual_lora.py --tasks 3 --n_shot 32 --lora_decoder \
        --pretrained_path "$PRET" --data_dir "$DATA" --epochs $EPOCHS \
        --seed $SEED --save_dir "$sd"
  done
}

# ============================================================================
# BATCH 4: All 6 methods — supplement seed 45, 46 (n_shot=32)
#   Reviewer: "增加 seeds 數量"
# ============================================================================
run_batch_4() {
  log "===== BATCH 4: All Methods seed 45,46 (n32) ====="

  for SEED in 45 46; do
    # LoRA
    local sd="./outputs/miccai_experiments/n32/lora/seed${SEED}"
    run_exp "LoRA n32 seed=$SEED" \
      "$(metrics_for "$sd" lora)" \
      $PY run_continual_lora.py --tasks 2 3 --n_shot 32 --lora_decoder \
        --pretrained_path "$PRET" --data_dir "$DATA" --epochs $EPOCHS \
        --seed $SEED --save_dir "$sd"

    # Sequential Linear
    sd="./outputs/miccai_experiments/n32/sequential_linear/seed${SEED}"
    run_exp "SeqLinear n32 seed=$SEED" \
      "$(metrics_for "$sd" sequential_linear)" \
      $PY run_continual_baselines.py --baseline sequential_linear --tasks 2 3 --n_shot 32 \
        --pretrained_path "$PRET" --data_dir "$DATA" --epochs $EPOCHS \
        --seed $SEED --save_dir "$sd"

    # Sequential FT
    sd="./outputs/miccai_experiments/n32/sequential_ft/seed${SEED}"
    run_exp "SeqFT n32 seed=$SEED" \
      "$(metrics_for "$sd" sequential_ft)" \
      $PY run_continual_baselines.py --baseline sequential_ft --tasks 2 3 --n_shot 32 \
        --pretrained_path "$PRET" --data_dir "$DATA" --epochs $EPOCHS \
        --seed $SEED --save_dir "$sd"

    # EWC
    sd="./outputs/miccai_experiments/n32/ewc/seed${SEED}"
    run_exp "EWC n32 seed=$SEED" \
      "$(metrics_for "$sd" ewc)" \
      $PY run_continual_ewc.py --tasks 2 3 --n_shot 32 \
        --pretrained_path "$PRET" --data_dir "$DATA" --epochs $EPOCHS \
        --seed $SEED --save_dir "$sd"

    # LwF
    sd="./outputs/miccai_experiments/n32/lwf/seed${SEED}"
    run_exp "LwF n32 seed=$SEED" \
      "$(metrics_for "$sd" lwf)" \
      $PY run_continual_lwf.py --tasks 2 3 --n_shot 32 \
        --pretrained_path "$PRET" --data_dir "$DATA" --epochs $EPOCHS \
        --seed $SEED --save_dir "$sd"

    # Replay
    sd="./outputs/miccai_experiments/n32/replay/seed${SEED}"
    run_exp "Replay n32 seed=$SEED" \
      "$(metrics_for "$sd" replay)" \
      $PY run_continual_replay.py --tasks 2 3 --n_shot 32 \
        --pretrained_path "$PRET" --data_dir "$DATA" --epochs $EPOCHS \
        --seed $SEED --save_dir "$sd"
  done
}

# ============================================================================
# BATCH 5: Shot ablation n_shot=16 — supplement seed 45, 46
#   Reviewer: "shot ablation 完整化"
# ============================================================================
run_batch_5() {
  log "===== BATCH 5: Shot Ablation n16 seed 45,46 ====="

  for SEED in 45 46; do
    local sd="./outputs/ablations/lora/seed${SEED}_shot16"
    run_exp "LoRA n16 seed=$SEED" \
      "$(metrics_for "$sd" lora)" \
      $PY run_continual_lora.py --tasks 2 3 --n_shot 16 --lora_decoder \
        --pretrained_path "$PRET" --data_dir "$DATA" --epochs $EPOCHS \
        --seed $SEED --save_dir "$sd"
  done
}

# ============================================================================
# Main dispatch
# ============================================================================
log "=========================================="
log "JBHI Revision Experiments — START"
log "Batch=$BATCH  DryRun=$DRY_RUN"
log "=========================================="

case "$BATCH" in
  1)   run_batch_1 ;;
  2)   run_batch_2 ;;
  3)   run_batch_3 ;;
  4)   run_batch_4 ;;
  5)   run_batch_5 ;;
  all)
    run_batch_1
    run_batch_2
    run_batch_3
    run_batch_4
    run_batch_5
    ;;
  *)
    echo "Unknown batch: $BATCH (use 1-5 or all)"
    exit 1
    ;;
esac

log "=========================================="
log "JBHI Revision Experiments — COMPLETE"
log "Log: $MAIN_LOG"
log "=========================================="
