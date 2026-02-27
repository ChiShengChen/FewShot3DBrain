# Code Structure

## Entry Points

| File | Purpose |
|------|---------|
| `run_continual_lora.py` | Proposed LoRA continual learning (encoder+decoder adapters) |
| `run_continual_baselines.py` | Sequential Linear (frozen backbone) / Sequential FT (full fine-tune) |
| `run_continual_ewc.py` | EWC baseline |
| `run_continual_lwf.py` | LwF baseline |
| `run_continual_replay.py` | Replay baseline |
| `run_baselines.py` | Per-task baselines (separate model per task) |
| `run_multi_seed.py` | Multi-seed orchestration |
| `run_ablations.py` | Shot count / LoRA rank ablations |
| `run_single_task.py` | Single-task training |

## Core (`src/`)

| File | Purpose |
|------|---------|
| `data.py` | `TASK_CONFIGS`, `FewShotFOMODataset`, `get_few_shot_dataloaders` |
| `models.py` | `TaskModel` (backbone + task head) |
| `backbone.py` | `unet_b` wrapper, pretrained loading |
| `lora.py` | LoRA layers for Conv3d |
| `train.py` | `train_task`, `evaluate` |
| `replay_buffer.py` | Replay buffer for Replay baseline |

## Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `prepare_plan_b_data.py` | Convert BraTS/IXI → preprocessed `.npy` format |
| `run_miccai_experiments.py` | Phase 1/2/3 experiment orchestration |
| `aggregate_miccai_results.py` | Aggregate metrics across seeds → `aggregate.json` |
| `download_champion_weights.sh` | Download FOMO pretrained weights |
| `setup_reproduction.sh` | One-shot setup (venv, deps, weights) |
| `visualize_segmentation.py` | Visualization utilities |
| `wilcoxon_brain_age.py` | Statistical test for age bias |
| `stack_figure.py` | Figure generation for paper |

## Data Flow

```
Raw (BraTS/IXI) → prepare_plan_b_data.py → data/preprocessed/Task00X_FOMOX/
                                                    ↓
                                    create_fewshot → Task00X_fewshot16/32/64/
                                                    ↓
                                    get_few_shot_dataloaders() → DataLoader
                                                    ↓
                                    run_continual_*.py → outputs/
                                                    ↓
                                    aggregate_miccai_results.py → aggregate.json
```

## Task Notation (Paper vs Code)

| Paper | Code (task_id) | Dataset |
|-------|----------------|---------|
| T1 | 2 | BraTS (segmentation) |
| T2 | 3 | IXI (regression) |

Task 1 (infarct) is skipped in current experiments (ISLES lacks healthy negatives).
