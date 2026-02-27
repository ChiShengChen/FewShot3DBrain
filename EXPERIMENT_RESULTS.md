# FewShot3DBrain Experiment Results

**Last updated:** 2026-02-25 (MICCAI t32_t3t2 Phase 3 T3→T2 results added)  
**Hardware:** NVIDIA RTX 3090 (24GB)  
**Purpose:** MICCAI 2026 continual learning comparison

---

## 1. Data

### 1.1 Sources & Preparation

| Task | Name | Source | Modalities | Format |
|------|------|--------|------------|--------|
| 1 | Task001_FOMO1 (Infarct) | ISLES 2022 (Zenodo 7153326) | 4ch: DWI, FLAIR, ADC, pad | `.npy` + `.txt` |
| 2 | Task002_FOMO2 (Tumor) | BraTS 2023 Glioma | 3ch: T2w, T2-FLAIR, T1c | `.npy` + `_seg.npy` |
| 3 | Task003_FOMO3 (Age) | IXI | 2ch: T1, T2 | `.npy` + `.txt` |

**Preprocessed path:** `data/preprocessed/`  
**Raw / intermediate:**  
- ISLES: `data/plan_b_raw/isles/`  
- BraTS: `scripts/data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/` (1251 cases)  
- IXI: `data/plan_b_raw/ixi/`

**Script:** `scripts/prepare_plan_b_data.py`

### 1.2 Task Configs (`src/data.py` TASK_CONFIGS)

| Task | task_type | modalities | output |
|------|-----------|------------|--------|
| 1 | classification | 4 | binary (2 classes) |
| 2 | segmentation | 3 | binary mask |
| 3 | regression | 2 | scalar (age) |

### 1.3 Few-Shot Setup

- **n_shot:** 16 / 32 / 64 (default 32)
- **val_ratio:** 0.2
- **patch_size:** 64×64×64 (LoRA/baselines) or 32×32×32 (run_baselines)
- **Normalization:** per-channel Z-score
- **Note:** Task 1 skipped in current runs (ISLES has no healthy negatives)

---

## 2. Model

### 2.1 Backbone

- **Architecture:** `unet_b` from FOMO baseline  
- **Pretrained:** `weights/fomo25_mmunetvae_pretrained.ckpt` (PyTorch Lightning)  
- **Loading:** 39/40 encoder keys for encoder-only (T1, T3); 59/82 keys for full UNet (T2)

### 2.2 Task Heads

| Task | Backbone mode | Head |
|------|---------------|------|
| 1 | encoder-only | ClsRegHead (2 classes) |
| 2 | full UNet (encoder + decoder) | built-in decoder |
| 3 | encoder-only | ClsRegHead (1 output) |

### 2.3 LoRA Config (Proposed)

- **r:** 8  
- **lora_alpha:** 16  
- **lora_dropout:** 0.1  
- **Target:** encoder only (default); encoder + decoder with `--lora_decoder`  
- **Implementation:** `src/lora.py` (custom Conv3d LoRA)

### 2.4 Baselines

- **Sequential Linear:** backbone frozen, only heads/decoder trained sequentially  
- **Sequential FT:** full fine-tuning sequentially (catastrophic forgetting baseline)

---

## 3. Experiment Scripts & Commands

### 3.1 Data Preparation

| Script | Purpose |
|--------|---------|
| `scripts/prepare_plan_b_data.py` | Convert ISLES/BraTS/IXI → preprocessed `.npy` format |

**Key arguments:**
| Arg | Default | Description |
|-----|---------|--------------|
| `--task` | - | 1, 2, or 3 |
| `--all` | - | Prepare all tasks |
| `--download` | - | Download ISLES/IXI from URLs |
| `--source_dir` | - | Path to raw BraTS/ISLES/IXI |
| `--out_dir` | `data/preprocessed` | Output dir |
| `--work_dir` | `data/plan_b_raw` | Raw download dir |
| `--create_fewshot` | - | Create fewshot_16/32/64 splits |
| `--n_shots` | 16 32 64 | Shot counts |
| `--add_ixi_healthy` | - | Add IXI healthy as Task 1 negatives |
| `--n_healthy` | 250 | Max IXI healthy samples |

**Example:**
```bash
python scripts/prepare_plan_b_data.py --task 2 --source_dir ./scripts/data/BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData --out_dir data/preprocessed
python scripts/prepare_plan_b_data.py --all --download --add_ixi_healthy --create_fewshot
```

---

### 3.2 run_continual_lora.py (Proposed LoRA)

**Purpose:** Sequential continual learning with task-specific LoRA adapters.

| Arg | Default | Description |
|-----|---------|--------------|
| `--data_dir` | `./data/preprocessed` | Preprocessed data path |
| `--save_dir` | `./outputs/continual_lora` | Output dir |
| `--pretrained_path` | None | FOMO checkpoint path |
| `--tasks` | 1 2 3 | Tasks to run (e.g. `2 3`) |
| `--n_shot` | 32 | Few-shot samples (16/32/64) |
| `--patch_size` | 64 | Spatial patch size |
| `--batch_size` | 2 | Batch size |
| `--epochs` | 100 | Epochs per task |
| `--lr` | 1e-3 | Learning rate |
| `--lora_r` | 8 | LoRA rank |
| `--lora_decoder` | False | Add LoRA to T2 decoder (default: encoder-only) |
| `--create_dummy` | - | Create dummy data |
| `--seed` | 42 | Random seed |

**Full command (encoder-only LoRA, T2→T3):**
```bash
python run_continual_lora.py \
  --tasks 2 3 \
  --pretrained_path ./weights/fomo25_mmunetvae_pretrained.ckpt \
  --data_dir ./data/preprocessed \
  --n_shot 32 \
  --epochs 100 \
  --lr 1e-3 \
  --lora_r 8
```

**Encoder+decoder LoRA (legacy):**
```bash
python run_continual_lora.py --tasks 2 3 --pretrained_path ./weights/fomo25_mmunetvae_pretrained.ckpt --lora_decoder
```

---

### 3.3 run_continual_baselines.py (Continual Baselines)

**Purpose:** One model, sequential T1→T2→T3. Sequential Linear or Sequential FT.

| Arg | Default | Description |
|-----|---------|--------------|
| `--baseline` | sequential_linear | `sequential_linear` or `sequential_ft` |
| `--data_dir` | `./data/preprocessed` | Preprocessed data path |
| `--save_dir` | `./outputs/continual_baselines` | Output dir |
| `--pretrained_path` | None | FOMO checkpoint path |
| `--tasks` | 1 2 3 | Tasks to run |
| `--n_shot` | 32 | Few-shot samples (16/32/64) |
| `--patch_size` | 64 | Spatial patch size |
| `--epochs` | 100 | Epochs per task |
| `--create_dummy` | - | Create dummy data |
| `--seed` | 42 | Random seed |

**Sequential Linear (frozen backbone):**
```bash
python run_continual_baselines.py \
  --baseline sequential_linear \
  --tasks 2 3 \
  --pretrained_path ./weights/fomo25_mmunetvae_pretrained.ckpt
```

**Sequential FT (full fine-tune, + T2 re-eval for forgetting):**
```bash
python run_continual_baselines.py \
  --baseline sequential_ft \
  --tasks 2 3 \
  --pretrained_path ./weights/fomo25_mmunetvae_pretrained.ckpt
```

---

### 3.4 run_baselines.py (Per-Task Baselines)

**Purpose:** Separate model per task (no continual, no shared backbone).

| Arg | Default | Description |
|-----|---------|--------------|
| `--baseline` | linear | `linear` or `sequential_ft` |
| `--data_dir` | `./data/preprocessed` | Preprocessed data path |
| `--save_dir` | `./outputs/baselines` | Output dir |
| `--pretrained_path` | None | FOMO checkpoint path |
| `--tasks` | 1 2 3 | Tasks to run |
| `--n_shot` | 32 | Few-shot samples |
| `--epochs` | 50 | Epochs per task |
| `--create_dummy` | - | Create dummy data |

**Note:** Uses `patch_size=32` (fixed in script), unlike continual scripts (64).

**Per-task Linear:**
```bash
python run_baselines.py \
  --baseline linear \
  --tasks 2 3 \
  --pretrained_path ./weights/fomo25_mmunetvae_pretrained.ckpt
```

**Per-task Sequential FT:**
```bash
python run_baselines.py --baseline sequential_ft --tasks 2 3 --pretrained_path ./weights/fomo25_mmunetvae_pretrained.ckpt
```

---

### 3.5 run_multi_seed.py (3 seeds + BWT/FWT)

**Purpose:** Run experiments with multiple seeds, compute mean±std and BWT.

| Arg | Default | Description |
|-----|---------|--------------|
| `--method` | all | `all`, `lora`, `sequential_linear`, `sequential_ft` |
| `--seeds` | 42 43 44 | Seeds to run |
| `--pretrained_path` | `./weights/...` | FOMO checkpoint |
| `--n_shot` | 32 | Few-shot samples |
| `--epochs` | 100 | Epochs per task |

**BWT** = R_3,2 − R_2,2 (T2 performance after T3 minus T2 right after T2). Negative = forgetting.

**Command:**
```bash
python run_multi_seed.py --method all --seeds 42 43 44 --pretrained_path ./weights/fomo25_mmunetvae_pretrained.ckpt
```

**Output:** `outputs/multi_seed/aggregate.json` with mean±std per metric and BWT.

---

### 3.7 run_ablations.py (Shot & LoRA Rank Ablations)

**Purpose:** Systematic ablations over shot count (16/32/64) and LoRA rank (4/8/16).

| Arg | Default | Description |
|-----|---------|-------------|
| `--ablations` | shot lora_rank | `shot`, `lora_rank`, or both |
| `--method` | all | `all`, `lora`, `sequential_linear`, `sequential_ft` |
| `--seeds` | 42 43 44 | Random seeds |
| `--pretrained_path` | `./weights/...` | FOMO checkpoint |
| `--epochs` | 100 | Epochs per task |

**Commands:**
```bash
python run_ablations.py --ablations shot --method all
python run_ablations.py --ablations lora_rank --method lora
```

**Output:** `outputs/ablations/<method>/seed<N>_shot<K>/metrics.json`, `outputs/ablations/ablation_summary.json`

---

### 3.8 Quick Reference

| Experiment | Script | Key Cmd |
|------------|--------|---------|
| LoRA (encoder-only) | run_continual_lora.py | `--tasks 2 3 --pretrained_path ...` |
| LoRA (enc+dec) | run_continual_lora.py | `--tasks 2 3 --pretrained_path ... --lora_decoder` |
| Sequential Linear | run_continual_baselines.py | `--baseline sequential_linear --tasks 2 3 ...` |
| Sequential FT | run_continual_baselines.py | `--baseline sequential_ft --tasks 2 3 ...` |
| Per-task Linear | run_baselines.py | `--baseline linear --tasks 2 3 ...` |
| Multi-seed + BWT | run_multi_seed.py | `--method all --seeds 42 43 44 ...` |
| Shot/LoRA ablations | run_ablations.py | `--ablations shot --method all` |

---

## 4. Results

### 4.1 Real Data (Plan B: BraTS + IXI)

**Tasks:** 2 → 3 (skip Task 1)

| Method | Script | T2 Dice↑ | T3 MAE↓ | Notes |
|--------|--------|----------|---------|-------|
| **LoRA enc+dec** | run_continual_lora.py | 0.50 | 0.20 | `--lora_decoder`, continual |
| **LoRA enc-only** | run_continual_lora.py | 0.19 | 0.20 | default, continual |
| **Sequential Linear** | run_continual_baselines.py | **0.76** | 1.49 | One model T2→T3, continual |
| **Per-task Linear** | run_baselines.py | 0.65 | **0.063** | Separate model per task, patch=32, epochs=50 |
| **Sequential FT** | run_continual_baselines.py | 0.78 | 0.002 | Full FT; **T2 after T3: 0.006** (Δ=0.77) |

### 4.2 LoRA Ablation: Encoder-only vs Encoder+Decoder

| LoRA Config | T2 Dice | T3 MAE |
|-------------|---------|--------|
| Encoder+decoder (`--lora_decoder`) | **0.50** | 0.20 |
| Encoder-only (default) | 0.19 | 0.20 |

*Encoder-only LoRA worsened T2 (0.19 vs 0.50); T3 similar. Segmentation needs decoder adaptation.*

### 4.3 Per-Epoch (LoRA enc+dec, real data)

**Task 2 (Segmentation):**
| Epoch | Val Dice | Val Loss |
|-------|----------|----------|
| 10 | 0.159 | 1.47 |
| 50 | 0.339 | 1.37 |
| 80 | **0.501** | 1.31 |
| 100 | 0.402 | 1.30 |

**Task 3 (Regression):**
| Epoch | Val MAE | Val Loss |
|-------|---------|----------|
| 20 | 0.47 | 0.09 |
| 50 | 0.22 | 0.01 |
| 100 | **0.20** | 0.002 |

### 4.4 Comparison Summary

| Metric | LoRA enc+dec | LoRA enc-only | Sequential Linear | Per-task Linear |
|--------|--------------|---------------|-------------------|-----------------|
| T2 Dice | 0.50 | 0.19 | **0.76** | 0.65 |
| T3 MAE | 0.20 | 0.20 | 1.49 | **0.063** |

- **T2:** Sequential FT (0.82) ≥ Sequential Linear (0.76) > Per-task Linear (0.65) > LoRA (0.50)
- **T3 (MAE↓):** Sequential FT 0.002 (likely overfit) / Per-task 0.063 / LoRA 0.20 / Sequential Linear 1.49
- **Sequential FT T3 MAE 0.002** may indicate severe overfitting on few-shot val
- **Catastrophic forgetting:** Need to re-eval T2 *after* T3 training to measure T2 performance drop

### 4.5 Sequential Linear T2→T3 (Real Data, Latest Run)

**Task 2:**
| Epoch | Val Dice | Val Loss |
|-------|----------|----------|
| 10 | 0.54 | 0.84 |
| 60 | 0.73 | 0.56 |
| 80 | **0.73** | 0.61 |
| 100 | 0.69 | 0.61 |

**Task 3:**
| Epoch | Val MAE | Val Loss |
|-------|---------|----------|
| 20 | 1.94 | 4.80 |
| 90 | **1.49** | 2.63 |
| 100 | 1.49 | 2.61 |

**Final:** T2 Dice 0.76, T3 MAE 1.49

*T1 AUC=0.5: ISLES lacks healthy negatives.*

### 4.6 Dummy Data (Sanity Check, 2 epochs)

| Method | T2 Dice | T3 MAE |
|--------|---------|--------|
| Sequential Linear (--create_dummy) | 0.91 | 45.6 |

### 4.7 Sequential FT T2→T3 + Catastrophic Forgetting

**Task 2:** Dice 0.82 (right after T2)  
**Task 3:** MAE 0.002 (likely overfit on few-shot)

**Re-eval T2 after T3** (encoder overwritten by T3):
| Metric | Value |
|--------|-------|
| T2 Dice (right after T2) | 0.778 |
| T2 Dice (after T3 training) | **0.006** |
| Forgetting Δ | **0.77** |

*Catastrophic forgetting clearly demonstrated: T2 performance collapses when backbone is fine-tuned for T3.*

### 4.8 Multi-Seed Results (3 seeds: 42, 43, 44)

| Method | T2 Dice | T3 MAE | T2 after T3 | BWT |
|--------|---------|--------|-------------|-----|
| **LoRA** (enc+dec) | 0.62 ± 0.07 | 0.16 ± 0.05 | (=T2) | 0.00 |
| **Sequential Linear** | 0.79 ± 0.006 | 1.45 ± 0.03 | 0.78 ± 0.006 | **-0.010 ± 0.006** |
| **Sequential FT** | 0.80 ± 0.02 | 0.004 ± 0.002 | 0.16 ± 0.16 | **-0.65 ± 0.14** |

*LoRA uses task-specific adapters, so BWT=0. Sequential Linear: mild forgetting after T3. Sequential FT: severe catastrophic forgetting.*

### 4.9 MICCAI Experiments (n32, seeds 42–44)

**Script:** `python scripts/run_miccai_experiments.py --phase 1`  
**Aggregate:** `python scripts/aggregate_miccai_results.py` → `outputs/miccai_experiments/n32/aggregate.json`

| Method | T1 Dice↑ | T2 MAE↓ | T1 after T2 | BWT |
|--------|---------|---------|-------------|-----|
| **LoRA** (enc+dec) | 0.60 ± 0.08 | 0.012 ± 0.003 | (=T1) | **0.00** |
| Sequential Linear | 0.79 ± 0.01 | 1.45 ± 0.03 | 0.78 ± 0.01 | -0.01 ± 0.01 |
| Sequential FT | 0.80 ± 0.02 | 0.005 ± 0.003 | 0.16 ± 0.19 | -0.65 ± 0.17 |
| EWC | 0.79 ± 0.02 | 0.001 ± 0.001 | 0.15 ± 0.24 | -0.65 ± 0.23 |
| LwF | 0.80 ± 0.02 | 0.020 ± 0.009 | 0.25 ± 0.22 | -0.56 ± 0.24 |
| Replay | 0.79 ± 0.01 | 0.021 ± 0.013 | 0.01 ± 0.01 | -0.78 ± 0.02 |

*LoRA: zero forgetting (BWT=0). EWC: lowest T2 MAE but severe forgetting. LwF/Replay: strong T1, moderate forgetting.*

### 4.10 MICCAI Experiments (n64, seeds 42–44)

**Script:** `python scripts/run_miccai_experiments.py --phase 2`  
**Aggregate:** `python scripts/aggregate_miccai_results.py n64` → `outputs/miccai_experiments/n64/aggregate.json`

| Method | T1 Dice↑ | T2 MAE↓ | T1 after T2 | BWT |
|--------|---------|---------|-------------|-----|
| EWC | 0.84 ± 0.02 | 0.001† ± 0.0003 | 0.07 ± 0.10 | -0.77 ± 0.11 |
| LwF | 0.82 ± 0.01 | 0.038 ± 0.014 | 0.25 ± 0.41 | -0.58 ± 0.42 |

*Phase 2 runs EWC and LwF only (no Replay). LoRA and sequential methods skipped (legacy complete). EWC T2 MAE likely overfits. LwF T1 after T2 has high variance.*

### 4.11 MICCAI Experiments (t32_t3t2, Phase 3, T3→T2 order, seeds 42–44)

**Script:** `python scripts/run_miccai_experiments.py --phase 3 --force`  
**Output:** `outputs/miccai_experiments/t32_t3t2/`  
**Aggregate:** `python scripts/aggregate_miccai_results.py t32_t3t2` → `outputs/miccai_experiments/t32_t3t2/aggregate.json`

Task order: T3 (regression) first, then T2 (segmentation). BWT = T2 MAE after T1 minus T2 MAE after T2 (positive = T3 forgetting).

| Method | T1 Dice↑ | T2 MAE↓ | T2 after T1 | BWT |
|--------|---------|---------|-------------|-----|
| LoRA (enc+dec) | 0.50 ± 0.08 | 0.005 ± 0.002 | --- | **0.00** |
| Sequential Linear | 0.78 ± 0.01 | 0.26 ± 0.003 | 0.36 ± 0.05 | 0.10 ± 0.05 |
| Sequential FT | 0.81 ± 0.01 | 0.001† ± 0.0003 | 7.17 ± 1.41 | 7.16 ± 1.41 |

*LoRA: zero forgetting. Sequential Linear: mild T3 forgetting (BWT≈0.10). Sequential FT: severe T3 forgetting (T2 MAE 0.001→7.17 after T1 training). Seq FT T2 MAE likely overfits.*

### 4.12 Saved Outputs

- **MICCAI n32 aggregate:** `outputs/miccai_experiments/n32/aggregate.json`
- **MICCAI n64 aggregate:** `outputs/miccai_experiments/n64/aggregate.json`
- **MICCAI t32 aggregate:** `outputs/miccai_experiments/t32/aggregate.json` (legacy T2→T3)
- **MICCAI t32_t3t2 aggregate:** `outputs/miccai_experiments/t32_t3t2/aggregate.json` (T3→T2 order)
- **MICCAI per-method:** `outputs/miccai_experiments/n32/<method>/seed*/metrics.json`, `outputs/miccai_experiments/n64/<method>/seed*/metrics.json`, `outputs/miccai_experiments/t32_t3t2/<method>/seed*/metrics.json`
- **Legacy aggregate:** `outputs/multi_seed/aggregate.json`
- **LoRA:** `outputs/multi_seed/lora/seed*/metrics.json`
- **Sequential Linear:** `outputs/multi_seed/sequential_linear/seed*/sequential_linear/metrics.json`
- **Sequential FT:** `outputs/multi_seed/sequential_ft/seed*/sequential_ft/metrics.json`
- **Ablations:** `outputs/ablations/<method>/seed<N>_shot<K>/metrics.json`

---

### 4.13 Shot Ablation (n_shot=16, LoRA enc+dec)

**Setting:** LoRA r=8, encoder+decoder, seeds 42–44. Task 2→3.

| Seed | T2 Dice | T3 MAE | T2 time (sec) | T3 time (sec) | GPU max (MB) |
|------|---------|--------|---------------|---------------|--------------|
| 42 | 0.427 | 0.338 | 3877 | 94 | 1670 |
| 43 | 0.482 | 0.322 | 25876 | 777 | 1670 |
| 44 | (running) | — | — | — | — |

**Mean (seed 42–43):** T2 Dice 0.45 ± 0.04, T3 MAE 0.33 ± 0.01

**vs n_shot=32 (multi-seed):** T2 0.62±0.07, T3 MAE 0.16±0.05 → 16-shot LoRA has lower T2 Dice and higher T3 MAE; few-shot sensitivity confirmed.

---

## 5. Notes & Next Steps

1. **Task 1:** Skipped; ISLES is mostly positive, needs IXI healthy negatives (`--add_ixi_healthy`).
2. **T2 LoRA:** Encoder-only (0.19) < encoder+decoder (0.50) < Sequential Linear (0.76). Encoder-only did not help.
3. **T3 LoRA:** Strong result; encoder + LoRA works well for regression.
4. **Sequential FT:** Run to quantify catastrophic forgetting.
5. **BWT/FWT:** Not yet computed; add evaluation on all previous tasks after each step.
