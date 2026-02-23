#!/bin/bash
#
# MICCAI 2026 experiments: baselines + n_shot=64 + task order ablation
# Run from project root: bash scripts/run_miccai_experiments.sh
#
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-./data/preprocessed}"
PRETRAINED="${PRETRAINED_PATH:-./weights/fomo25_mmunetvae_pretrained.ckpt}"
SEEDS="42 43 44"
EPOCHS=100

# Build pretrained args if file exists
PRET_ARGS=()
[[ -f "$PRETRAINED" ]] && PRET_ARGS=(--pretrained_path "$PRETRAINED")

echo "=== FewShot3DBrain MICCAI Experiments ==="
echo "DATA_DIR=$DATA_DIR"
echo "PRETRAINED=$PRETRAINED"
echo ""

# --- 1. Main comparisons: n_shot=32 (T2→T3) ---
echo ">>> Phase 1: Main comparison (n_shot=32, T2→T3)"
for method in lora sequential_linear sequential_ft ewc lwf replay; do
    for seed in $SEEDS; do
        echo "  $method seed=$seed"
        case $method in
            lora)
                python run_continual_lora.py --tasks 2 3 --data_dir "$DATA_DIR" --save_dir outputs/miccai_experiments/n32/$method/seed$seed \
                    --n_shot 32 --epochs $EPOCHS --seed $seed --lora_decoder "${PRET_ARGS[@]}"
                ;;
            sequential_linear|sequential_ft)
                python run_continual_baselines.py --baseline $method --tasks 2 3 --data_dir "$DATA_DIR" \
                    --save_dir outputs/miccai_experiments/n32/$method/seed$seed \
                    --n_shot 32 --epochs $EPOCHS --seed $seed "${PRET_ARGS[@]}"
                ;;
            ewc)
                python run_continual_ewc.py --tasks 2 3 --data_dir "$DATA_DIR" --save_dir outputs/miccai_experiments/n32/ewc/seed$seed \
                    --n_shot 32 --epochs $EPOCHS --seed $seed "${PRET_ARGS[@]}"
                ;;
            lwf)
                python run_continual_lwf.py --tasks 2 3 --data_dir "$DATA_DIR" --save_dir outputs/miccai_experiments/n32/lwf/seed$seed \
                    --n_shot 32 --epochs $EPOCHS --seed $seed "${PRET_ARGS[@]}"
                ;;
            replay)
                python run_continual_replay.py --tasks 2 3 --data_dir "$DATA_DIR" --save_dir outputs/miccai_experiments/n32/replay/seed$seed \
                    --n_shot 32 --epochs $EPOCHS --seed $seed "${PRET_ARGS[@]}"
                ;;
        esac
    done
done

# --- 2. n_shot=64: scaling behavior ---
echo ""
echo ">>> Phase 2: n_shot=64 (scaling)"
for method in lora sequential_linear sequential_ft ewc lwf; do
    for seed in $SEEDS; do
        echo "  $method n64 seed=$seed"
        case $method in
            lora)
                python run_continual_lora.py --tasks 2 3 --data_dir "$DATA_DIR" --save_dir outputs/miccai_experiments/n64/$method/seed$seed \
                    --n_shot 64 --epochs $EPOCHS --seed $seed --lora_decoder "${PRET_ARGS[@]}"
                ;;
            sequential_linear|sequential_ft)
                python run_continual_baselines.py --baseline $method --tasks 2 3 --data_dir "$DATA_DIR" \
                    --save_dir outputs/miccai_experiments/n64/$method/seed$seed \
                    --n_shot 64 --epochs $EPOCHS --seed $seed "${PRET_ARGS[@]}"
                ;;
            ewc)
                python run_continual_ewc.py --tasks 2 3 --data_dir "$DATA_DIR" --save_dir outputs/miccai_experiments/n64/ewc/seed$seed \
                    --n_shot 64 --epochs $EPOCHS --seed $seed "${PRET_ARGS[@]}"
                ;;
            lwf)
                python run_continual_lwf.py --tasks 2 3 --data_dir "$DATA_DIR" --save_dir outputs/miccai_experiments/n64/lwf/seed$seed \
                    --n_shot 64 --epochs $EPOCHS --seed $seed "${PRET_ARGS[@]}"
                ;;
        esac
    done
done

# --- 3. Task order ablation: T3→T2 (IXI before BraTS) ---
echo ""
echo ">>> Phase 3: Task order T3→T2"
for method in lora sequential_linear sequential_ft; do
    for seed in $SEEDS; do
        echo "  $method T3→T2 seed=$seed"
        case $method in
            lora)
                python run_continual_lora.py --tasks 3 2 --data_dir "$DATA_DIR" --save_dir outputs/miccai_experiments/t32/$method/seed$seed \
                    --n_shot 32 --epochs $EPOCHS --seed $seed --lora_decoder "${PRET_ARGS[@]}"
                ;;
            sequential_linear|sequential_ft)
                python run_continual_baselines.py --baseline $method --tasks 3 2 --data_dir "$DATA_DIR" \
                    --save_dir outputs/miccai_experiments/t32/$method/seed$seed \
                    --n_shot 32 --epochs $EPOCHS --seed $seed "${PRET_ARGS[@]}"
                ;;
        esac
    done
done

echo ""
echo "=== Done. Results in outputs/miccai_experiments/ ==="
