# Reproduction Guide (MICCAI 2026)

This document describes how to reproduce the experiments in the paper.

## Environment

- **Python:** 3.9+ (3.10 recommended)
- **GPU:** NVIDIA GPU with ≥8GB VRAM (tested on RTX 3090 24GB)
- **OS:** Linux / macOS (Windows should work with minor path adjustments)

## Step 1: Clone and Setup

```bash
git clone https://github.com/<your-repo>/FewShot3DBrain.git
cd FewShot3DBrain

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Step 2: Download Pretrained Weights

```bash
chmod +x scripts/download_champion_weights.sh
./scripts/download_champion_weights.sh
```

This downloads `weights/fomo25_mmunetvae_pretrained.ckpt` (~100MB) from [jbanusco/fomo25](https://github.com/jbanusco/fomo25).

## Step 3: Prepare Data

### Option A: Full Data (BraTS + IXI)

**Task 2 (BraTS 2023 Glioma):**
- Register at [Synapse](https://www.synapse.org/) and accept the Data Use Agreement
- Download from [syn51156910](https://www.synapse.org/#!Synapse:syn51156910)
- Extract and run:
```bash
python scripts/prepare_plan_b_data.py --task 2 \
  --source_dir /path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
  --out_dir data/preprocessed
```

**Task 3 (IXI):**
```bash
python scripts/prepare_plan_b_data.py --task 3 --download --out_dir data/preprocessed
# Creates data/plan_b_raw/ixi/ and data/preprocessed/Task003_FOMO3/
```

**Create few-shot splits:**
```bash
python scripts/prepare_plan_b_data.py --create_fewshot --n_shots 16 32 64 \
  --out_dir data/preprocessed
```

### Option B: Dummy Data (Quick Sanity Check)

Use `--create_dummy` in the experiment commands below. No data preparation needed.

## Step 4: Run Experiments

All experiments use seeds 42, 43, 44 by default.

### Phase 1: n_shot=32, T2→T3 (Main Results)

```bash
python scripts/run_miccai_experiments.py --phase 1
```

- Methods: LoRA, Sequential Linear, Sequential FT, EWC, LwF, Replay
- Output: `outputs/miccai_experiments/n32/`
- Approx. time: ~12–24 hours (depends on GPU)

### Phase 2: n_shot=64

```bash
python scripts/run_miccai_experiments.py --phase 2
```

- Methods: LoRA, Sequential Linear, Sequential FT, EWC, LwF
- Output: `outputs/miccai_experiments/n64/`

### Phase 3: Task Order T3→T2

```bash
python scripts/run_miccai_experiments.py --phase 3 --force
```

- Methods: LoRA, Sequential Linear, Sequential FT
- Output: `outputs/miccai_experiments/t32_t3t2/`

### With Dummy Data (Fast Test)

```bash
python scripts/run_miccai_experiments.py --phase 1 --create_dummy --epochs 5
```

## Step 5: Aggregate Results

```bash
# Phase 1
python scripts/aggregate_miccai_results.py n32

# Phase 2
python scripts/aggregate_miccai_results.py n64

# Phase 3 (T3→T2)
python scripts/aggregate_miccai_results.py t32_t3t2
```

Output: `outputs/miccai_experiments/<phase>/aggregate.json`

## Expected Results (Phase 1, n32)

| Method            | T1 Dice↑    | T2 MAE↓     | BWT   |
|-------------------|-------------|-------------|-------|
| LoRA (enc+dec)    | 0.60 ± 0.08 | 0.012 ± 0.003 | 0.00 |
| Sequential Linear | 0.79 ± 0.01 | 1.45 ± 0.03  | -0.01 |
| Sequential FT     | 0.80 ± 0.02 | 0.005† ± 0.003 | -0.65 |
| EWC               | 0.79 ± 0.02 | 0.001† ± 0.001 | -0.65 |
| LwF               | 0.80 ± 0.02 | 0.020 ± 0.009 | -0.56 |
| Replay            | 0.79 ± 0.01 | 0.021 ± 0.013 | -0.78 |

† EWC and Sequential FT T2 MAE likely overfit on few-shot validation.

## Script Reference

| Script | Purpose |
|--------|---------|
| `scripts/run_miccai_experiments.py` | Run all phases (orchestrator) |
| `scripts/aggregate_miccai_results.py` | Aggregate metrics across seeds |
| `run_continual_lora.py` | LoRA continual learning |
| `run_continual_baselines.py` | Sequential Linear / FT |
| `run_continual_ewc.py` | EWC baseline |
| `run_continual_lwf.py` | LwF baseline |
| `run_continual_replay.py` | Replay baseline |

## Troubleshooting

**CUDA out of memory:** Reduce batch size in `src/train.py` or `run_continual_*.py` (default 2).

**Missing pretrained weights:** Ensure `weights/fomo25_mmunetvae_pretrained.ckpt` exists. Run `./scripts/download_champion_weights.sh`.

**IXI bad shapes:** IXI NIfTI affine can be wrong. The script uses `header.get_zooms()`. If you see `(2, 0, 30, 0)`, re-run:
```bash
python scripts/prepare_plan_b_data.py --task 3 --source_dir data/plan_b_raw/ixi
```

**BraTS path:** Task 2 expects BraTS 2023 Glioma structure: `.../ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/` with subdirs per case containing `*t2.nii.gz`, `*flair.nii.gz`, `*t1ce.nii.gz`, `*seg.nii.gz`.
