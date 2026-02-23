# MICCAI 2026 Paper Plan: Few-Shot Continual Learning for 3D Brain MRI with Frozen Foundation Models

**Deadline:** Feb 26, 2026 | **Schedule:** 14 days (Feb 13–26) | **Hardware:** 1× NVIDIA RTX 3090 (24GB)

---

## 1. Detailed Method Section Outline

### 1.1 Problem Formulation

**Continual learning setting:**
- **Input:** Pretrained backbone \(f_\theta\) (FOMO-60K weights), frozen. Sequential arrival of tasks \(\mathcal{T}_1, \mathcal{T}_2, \mathcal{T}_3\) with limited labeled data.
- **Tasks:** (1) Infarct detection (binary classification), (2) Meningioma segmentation (3D binary mask), (3) Brain age estimation (regression).
- **Few-shot:** \(K \in \{16, 32, 64\}\) labeled examples per task, sampled from FOMO 20% pre-evaluation subset.
- **Constraint:** No replay buffer; no access to previous-task data when training new tasks.
- **Goal:** Minimize catastrophic forgetting while maintaining/improving forward transfer. Learn \(\phi_k\) (task-specific parameters) such that performance on all seen tasks remains high after training on \(\mathcal{T}_k\).

**Notation:**
- \(f_\theta\): frozen encoder (and decoder for segmentation) from pretrained UNet.
- \(\phi_k\): LoRA adapter for task \(k\).
- \(h_k(\cdot)\): task head (classifier / decoder / regressor) for task \(k\).

---

### 1.2 LoRA Adaptation for 3D Architectures

**Core idea:** Inject low-rank matrices into linear layers of the backbone. For a weight matrix \(W \in \mathbb{R}^{d \times k}\):
\[
W' = W + \Delta W, \quad \Delta W = B \cdot A, \quad A \in \mathbb{R}^{r \times k}, \, B \in \mathbb{R}^{d \times r}
\]
with rank \(r \ll \min(d,k)\). Only \(A, B\) are trained.

**3D UNet specifics (FOMO baseline: `unet_b`, `unet_xl`):**
- **Target layers:** (a) 3D convolutions → treat as linear over flattened spatial dims, or (b) linear layers in attention/blocks. Practical choice: inject into **2D/3D Conv layers** via 1×1×1 convolutions (equivalent to linear in channel dim) for memory efficiency.
- **HuggingFace PEFT:** Use `LoraConfig` with `target_modules` mapped to UNet block names. For custom 3D UNets, implement `peft.tuners.lora.layer.LoraLayer` wrapping `nn.Conv3d` or use `get_peft_model()` with custom target module names.
- **Rank \(r\):** Start with \(r \in \{4, 8, 16\}\); aim for <0.1% trainable params. For `unet_b` (~50M params), 0.1% ≈ 50K params → ~4–8 LoRA modules with \(r=8\) typically suffice.

**Implementation sketch (PEFT):**
```python
from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["down_blocks.*.conv", "up_blocks.*.conv"],
    lora_dropout=0.1, bias="none", task_type=TaskType.FEATURE_EXTRACTION
)
model = get_peft_model(backbone, lora_config)
```

---

### 1.3 Continual Learning Mechanism

**Adapter isolation:**
- Each task \(k\) gets a **dedicated LoRA adapter** \(\phi_k\). When training on \(\mathcal{T}_k\), only \(\phi_k\) (and task head \(h_k\)) are updated; \(\phi_{<k}\) remain frozen.
- At inference for task \(k\): load backbone + \(\phi_k\) + \(h_k\). No shared adapter across tasks.

**Optional: prototype-based task routing (for future work / ablations):**
- Maintain prototype \(\mathbf{c}_k\) per task (mean of frozen backbone features on task \(k\) support set).
- At inference: compute distance to prototypes, route to closest task’s adapter + head. Useful if task identity is unknown (open-set setting); may not be needed for main results if task ID is given.

**Training protocol per task:**
1. **Task 1 (Infarct):** Freeze backbone, add LoRA(\(\phi_1\)) + classifier head. Train on 16/32/64 samples. Save \(\phi_1\), \(h_1\).
2. **Task 2 (Meningioma):** Freeze backbone + \(\phi_1\). Add LoRA(\(\phi_2\)) + segmentation head. Train on few-shot samples. Save \(\phi_2\), \(h_2\).
3. **Task 3 (Brain age):** Freeze backbone + \(\phi_1\), \(\phi_2\). Add LoRA(\(\phi_3\)) + regressor head. Train on few-shot samples. Save \(\phi_3\), \(h_3\).

**Evaluation:** After each task, evaluate on *all* previous tasks (no access to their data). Report Task1-acc, Task2-Dice, Task3-MAE, plus BWT/FWT.

---

## 2. Complete Experiment Plan

### 2.1 Baselines

| Baseline | Description | Implementation |
|----------|-------------|----------------|
| **Independent FT** | Train separate full fine-tuned models per task (oracle upper bound) | One model per task, full FT |
| **Sequential FT** | Train one model sequentially on T1→T2→T3, full fine-tuning | `run_continual_baselines.py --baseline sequential_ft` |
| **Sequential Linear** | Same as Linear probing but one model, sequential T1→T2→T3 | `run_continual_baselines.py --baseline sequential_linear` |
| **Linear probing** | Freeze backbone, train only task heads (per-task models) | `run_baselines.py --baseline linear` |
| **EWC** | Sequential FT + Elastic Weight Consolidation penalty | Add \(\lambda \sum_i F_i (\theta_i - \theta^*_i)^2\) |
| **LwF** | Learning without Forgetting (distillation from previous model) | Knowledge distillation loss on current backbone outputs |
| **Proposed (LoRA)** | Frozen backbone + task-specific LoRA adapters | As in §1 |

### 2.2 Ablation Studies

| Ablation | Levels | Purpose |
|----------|--------|---------|
| Adapter type | LoRA vs Adapter (bottleneck) vs (full FT) | Validate LoRA choice |
| Shot count | 16, 32, 64 | Few-shot sensitivity |
| Task order | T1→T2→T3 vs T2→T1→T3 vs T3→T1→T2 | Order robustness |
| LoRA rank | r=4, 8, 16 | Capacity vs overfitting |
| LoRA placement | Encoder only vs encoder+decoder (segmentation) | Where to adapt |

### 2.3 Evaluation Metrics

| Metric | Tasks | Formula/Definition |
|--------|-------|---------------------|
| **Per-task performance** | T1, T2, T3 | AUC (infarct), Dice (meningioma), MAE (brain age) |
| **Average accuracy** | All | Mean of normalized metrics across tasks |
| **BWT** (Backward Transfer) | All | \(\frac{1}{T-1}\sum_{k=1}^{T-1} R_{T,k} - R_{k,k}\) (performance on task \(k\) after training up to \(T\)) |
| **FWT** (Forward Transfer) | All | \(\frac{1}{T-1}\sum_{k=2}^{T} R_{k-1,k} - R_{0,k}\) (performance on task \(k\) before seeing it) |
| **GPU memory** | Training | Peak VRAM (GB) during training |
| **Training time** | Per task | Wall-clock minutes per task |

---

## 3. Day-by-Day 14-Day Implementation Schedule

| Day | Date | Focus | Deliverables |
|-----|------|-------|--------------|
| **1** | Feb 13 | Codebase setup | Clone FOMO baseline, install deps (PyTorch, PEFT, nibabel, etc.), verify FOMO pretrained weights load on RTX 3090 |
| **2** | Feb 14 | Data pipeline | Preprocess FOMO 20% subset for T1,T2,T3; implement dataloaders for 16/32/64 shot sampling; verify loaders |
| **3** | Feb 15 | LoRA integration | Map FOMO UNet to PEFT; implement LoRA for 3D Conv; verify forward pass, param count <0.1% |
| **4** | Feb 16 | Task heads + single-task LoRA | Implement classifier (T1), segmentation decoder (T2), regressor (T3); run 64-shot single-task LoRA for each |
| **5** | Feb 17 | Sequential LoRA (proposed) | Implement adapter isolation, sequential training T1→T2→T3; save/load adapters; run full pipeline 64-shot |
| **6** | Feb 18 | Baselines 1/2 | Implement & run Linear probing, Independent FT (64-shot) |
| **7** | Feb 19 | Baselines 3/4 | Implement & run Sequential FT, EWC; tune EWC λ |
| **8** | Feb 20 | Baseline 5 + metrics | Implement LwF; compute BWT, FWT, avg accuracy for all methods |
| **9** | Feb 21 | Ablations | LoRA rank (4,8,16); shot count (16,32,64); adapter type (LoRA vs Adapter) |
| **10** | Feb 22 | Ablations + task order | Task order (3 permutations); LoRA placement (enc only vs enc+dec) |
| **11** | Feb 23 | Writing & figures | Draft intro, method, results; create framework diagram, forgetting curves, comparison table layout |
| **12** | Feb 24 | Writing & tables | Complete results tables; ablation tables; GPU memory & time |
| **13** | Feb 25 | Polish & backup | Related work, abstract, discussion; format for MICCAI; backup submission |
| **14** | Feb 26 | Final submission | Final proofread, PDF generation, submit before deadline |

---

## 4. Risk Analysis & Fallback Plans

| Risk | Likelihood | Impact | Fallback |
|------|------------|--------|----------|
| **FOMO baseline incompatible with PEFT** | Medium | High | Manually wrap Conv3d with LoRA-like layers (custom `nn.Module`), skip `get_peft_model` |
| **24GB OOM on segmentation (T2)** | High | High | Reduce patch size (96→64), batch_size=1, gradient checkpointing; use 2D slices if 3D fails |
| **20% subset too small or missing** | Low | High | **Plan B:** 使用公開資料集（ISLES 2022、BraTS Meningioma/Glioma、IXI），執行 `scripts/prepare_plan_b_data.py --all --download` |
| **Baseline runs too slow** | Medium | Medium | Reduce epochs (500→200), fewer `train_batches_per_epoch`; prioritize proposed + 2 baselines if time-critical |
| **LoRA yields worse than linear probing** | Low | High | Ablation: try Adapter (bottleneck), or LoRA on encoder only with larger rank |
| **Task order effects dominate** | Medium | Low | Report all orders; emphasize “on average” in main result |
| **MICCAI template/latex issues** | Low | Medium | Use official template early (Day 1); test build on Day 11 |

---

## 5. Figure Plan (3–4 Figures for 8-Page Paper)

| Figure | Content | Page |
|--------|---------|------|
| **Fig 1: Framework** | Left: frozen FOMO backbone. Right: task-specific LoRA adapters (T1,T2,T3) with task heads. Arrows: data flow; “frozen” vs “trainable” annotations. | §3 |
| **Fig 2: Forgetting curves** | X-axis: task index (1,2,3); Y-axis: performance (AUC/Dice/MAE). Lines: Sequential FT (catastrophic drop), Linear probe, Proposed LoRA. After each task, show performance on all previous tasks. | §4 |
| **Fig 3: Main comparison** | Bar chart or table: Avg accuracy, BWT, FWT for Independent FT, Seq FT, Linear, EWC, LwF, Proposed. One panel per shot count (16/32/64) or one summary. | §4 |
| **Fig 4: Ablations** | (a) LoRA rank vs performance; (b) shot count vs performance; (c) task order robustness (box plot or small multi-bar). | §4 |

**Table layout (main results):**
| Method | T1 AUC↑ | T2 Dice↑ | T3 MAE↓ | Avg↑ | BWT↑ | FWT↑ | Params | GPU (GB) |
|--------|---------|----------|---------|------|------|------|--------|----------|

---

## 6. Related Work Structure

### (a) Brain MRI Foundation Models
- FOMO25 Challenge & FOMO-60K dataset (official website, HuggingFace, arXiv:2506.14432)
- AMAES: Augmented Masked Autoencoder (Munk et al., ADSMI 2024) — FOMO baseline pretraining
- UNETR, Swin UNETR for 3D medical imaging
- Self-supervised pretraining for brain MRI (Bao et al., He et al., Chen et al. — MAE variants)
- Med-BERT, RadImageNet (if discussing broader medical FMs)

### (b) Parameter-Efficient Fine-Tuning
- LoRA (Hu et al., ICLR 2022)
- Adapter modules (Houlsby et al., ICML 2019)
- PEFT library (HuggingFace)
- Medical imaging PEFT: LoRA for chest X-ray, pathology (cite recent Med-LoRA / domain-specific works)

### (c) Continual Learning in Medical Imaging
- EWC (Kirkpatrick et al., PNAS 2017)
- LwF (Li & Hoiem, ECCV 2016)
- Continual learning for medical segmentation (e.g., Ouyang et al., Liu et al.)
- Task-incremental / class-incremental in medical AI

---

## 7. Implementation Checklist (Actionable)

### Environment
- [ ] Python 3.10+, PyTorch 2.x, CUDA 11.8+
- [ ] `pip install peft transformers nibabel numpy torchio` (if needed)
- [ ] `git clone https://github.com/fomo25/baseline-codebase`
- [ ] **Pretrained:** 冠軍權重 [jbanusco/fomo25 releases](https://github.com/jbanusco/fomo25/releases/tag/v1.0.0) (`fomo25_mmunetvae_pretrained.ckpt`) 或 Docker `jbanusco/sslmmunetave:1.0.0`

### Data
- [ ] Download FOMO-60K 20% pre-evaluation subset (or full eval data)
- [ ] Run `preprocess.py` for pretrain; `run_preprocessing.py` for task 1,2,3
- [ ] Implement `FewShotSampler(n_shot=16/32/64)` for each task

**Plan B（FOMO 下游資料不可用時）：** 使用公開資料集，執行 `scripts/prepare_plan_b_data.py`：
- **Task 1：** ISLES 2022 (Zenodo) — FLAIR/DWI/ADC，分類標籤由分割 mask 推得
- **Task 2：** BraTS Meningioma 或 BraTS Glioma — T2/FLAIR/T1ce 三通道，二值分割
- **Task 3：** IXI — T1/T2，年齡由 IXI.xls 取得

### Model
- [ ] Load pretrained `unet_b` or `unet_xl`; verify forward pass
- [ ] Add LoRA via PEFT or custom wrapper
- [ ] Implement task heads: `nn.Linear` (T1), UNet decoder (T2), `nn.Linear` (T3)

### Training
- [ ] Single-task LoRA training loop (CE loss T1, Dice/BCE T2, MSE T3)
- [ ] Sequential loop with adapter save/load (`run_continual_lora.py`)
- [ ] **Continual baselines** (one model T1→T2→T3): `run_continual_baselines.py --baseline sequential_linear` / `sequential_ft`
- [ ] Per-task baselines: `run_baselines.py --baseline linear` / `sequential_ft`
- [ ] EWC, LwF (optional)

### Evaluation
- [ ] AUC (sklearn), Dice (numpy), MAE
- [ ] BWT, FWT computation
- [ ] `torch.cuda.max_memory_allocated()` for GPU memory

---

*Last updated: Feb 12, 2026. Good luck with the submission!*
