# Protocol Audit — what invalidates the current results

**Date:** 2026-09-05
**Branch:** `fix/eval-protocol`
**Scope:** everything reported in `EXPERIMENT_RESULTS.md` and `IEEE_JBHI_3dBrainLoRA/main.tex`

---

## Summary

Eight defects were found. Together they invalidate **every regression number in the paper**,
the reason Task 1 was dropped, the segmentation numbers' comparability, and the premise
that a shared frozen pretrained backbone is used across the task sequence.

| # | Defect | What it invalidates | Status |
|---|--------|---------------------|--------|
| 1 | Scalar labels discarded during cropping | All T2/T3 regression results; Task 1 AUC | **fixed** |
| 2 | IXI age join fails for zero-padded IDs | 83 age labels; the "impute to 50.0" limitation | **fixed** |
| 3 | No test set; random-crop, globally-pooled Dice | All reported Dice; all model selection | **fixed (infra)** |
| 4 | Only the first task loads pretrained weights | The "shared frozen backbone" premise | **fixed (new driver)** |
| 5 | Parameter-efficiency figures are wrong | "<0.1% trainable", "~50M backbone" (7 places) | **measured** |
| 6 | Baseline trains 180x more parameters than LoRA | The "LoRA underperforms" framing | **reported per row** |

| 7 | 219 unlabelled BraTS cases written as empty masks | 17% of every segmentation score | **fixed** |
| 8 | `np.loadtxt(dtype=int64)` is a hard error on numpy 2.x | Would crash Task 1 on the real env | **fixed** |

Two further data problems are **flagged, not fixed** — they need a decision (see the end).

---

## 1. Scalar labels never reached the model

`src/data.py::FewShotFOMODataset.__getitem__` cropped the volume with

```python
vol, label = self._random_crop(vol, label if self.task_type == "segmentation" else None)
```

`_random_crop` returns its `label` argument unchanged, so for classification and
regression it returned `None` — overwriting the real label loaded two lines earlier. The
fallback then substituted a constant:

```python
elif self.task_type == "regression":
    if label is None:
        label = 50.0        # every subject
```

**Evidence.** Same split, same seed, before the fix:

```
labels on disk        : mean 50.88, std 14.26, range 20.2–86.2
_load_label()         : [62.74, 68.51, 68.13, 55.64, 32.58, ...]   correct
__getitem__()         : [50.0,  50.0,  50.0,  50.0,  50.0,  ...]   constant
```

Re-running the released `n32/lora/seed42` checkpoint reproduces `MAE = 0.0140` with
predictions of 49.99 / 50.03 / 50.01 — the model learned to emit the constant.

**Consequences for the manuscript.** Every one of these claims rests on the constant:

- "LoRA attains the best T2 MAE (0.012) among continual methods" — the metric is degenerate.
- EWC 0.001 and Seq FT 0.005 flagged as "likely overfits on few-shot validation" — they
  are not overfitting; all three methods are regressing to a constant.
- "Sequential Linear fails on T2 (MAE 1.45)" — it is merely slower to converge to 50.0.
- Phase 3 (T3→T2): "Seq FT shows severe T3 forgetting, MAE 0.001 → 7.17" — this measures
  drift away from a constant, not forgetting.
- "Wilcoxon p<0.001 shows systematic age underestimation" — the test ran on residuals of
  order 1e-2 years around the constant.
- The "IXI age imputation" limitation, and the observation that the six best T2 samples all
  have GT 50.0 — a symptom of this bug, not of the imputation.

Classification took the same path (`label = 0` for every case), so the reported
`T1 AUC = 0.5` and the stated reason for dropping Task 1 ("ISLES lacks healthy negatives")
are also artefacts. The prepared data is in fact balanced 250/250.

**Fix.** Crop the label only for segmentation. Guarded by
`tests/test_data_integrity.py::test_regression_labels_survive_cropping`.

---

## 2. IXI age join silently failed for 83 subjects

`scripts/prepare_plan_b_data.py` matched subject IDs as strings. Filenames zero-pad
(`IXI002` → `"002"`) while `IXI.xls` stores `IXI_ID` as an integer (`2`), so every subject
below ID 100 missed and fell through to `or 50.0`.

```
577 label files: 83 wrong, 15 with no spreadsheet row, 3 with NaN AGE
recovered: 559 usable subjects, age mean 48.68, std 16.46, range 20.0–86.3
```

**Fix.** Match on `int`; write NaN when there is no usable age instead of imputing.
`list_case_files` now drops non-finite scalar labels, so unusable subjects are excluded
rather than trained on. `scripts/fix_ixi_labels.py` repairs the existing preprocessed set
in place (report: `data/preprocessed/ixi_label_repair_report.json`).

The manuscript's "IXI age imputation" caveat should be replaced by an exclusion statement:
559 of 577 subjects have a usable age; 18 are excluded.

---

## 3. Evaluation protocol

Three separate problems, all in the direction of optimistic, non-comparable numbers.

**No held-out test set.** `train_task` selected the best of 100 epochs on the validation
split and reported that maximum. The reported number is therefore a best-of-100 order
statistic on the scoring set, not a held-out estimate.

**Random-crop evaluation.** The validation dataset used the same random 64³ crop as
training, so each epoch scored a different, randomly chosen sub-volume — evaluation was
not reproducible, and most of each 240×240×155 volume was never seen.

**Globally-pooled Dice.** `evaluate` concatenated every voxel in the split into one Dice.
BraTS results are conventionally per-case Dice (plus HD95); the two are not comparable, and
pooling lets large tumours dominate.

**Subject leakage.** Splitting was per file, but BraTS ships longitudinal timepoints —
1470 volumes come from 1332 subjects, and 138 subjects have more than one timepoint. A
per-file split puts the same patient in train and test.

**Fix.** `src/splits.py` builds subject-grouped three-way splits (train = exact shot budget,
val = epoch selection only, test = reported, capped at 250 subjects), persisted to JSON so
every method is scored on identical cases. `src/eval.py` performs Gaussian-weighted
sliding-window inference over full volumes and reports per-case Dice + HD95; scalar tasks
average a fixed deterministic patch grid. Validation cropping is now centre-based
(`deterministic=True`) so epoch selection is not noise-driven.

For regression the new report includes R², Pearson r, bias and calibration slope — a
constant predictor scores R² ≈ 0 and slope ≈ 0, so defect 1 cannot recur unnoticed.

---

## 4. Only the first task used the pretrained backbone

`run_continual_lora.py:91`:

```python
use_pretrained = (task_id == tasks[0]) and args.pretrained_path
model = TaskModel(..., pretrained_path=args.pretrained_path if use_pretrained else None, ...)
```

`TaskModel(pretrained_path=None)` randomly initialises the backbone (verified: weights
differ from the loaded checkpoint). With `--tasks 2 3`, task 3's adapter was therefore
trained on **random** weights, not the frozen FOMO backbone.

This undercuts the paper's central setup ("a frozen FOMO-style pretrained backbone" shared
across the sequence) and also explains the reported equality between continual LoRA and
per-task LoRA: under this code path they are nearly the same computation.

**Fix.** `scripts/run_benchmark_v2.py` loads the pretrained checkpoint for every task, and
carries the backbone forward explicitly for the sequential baselines.

---

## 5. The parameter-efficiency numbers are wrong

The manuscript states "<0.1\% trainable parameters per task" and a "~50M param" backbone in
the abstract, introduction, Section III-B, the discussion and the conclusion. Measured:

| Task | Trainable | Total | Share |
|------|-----------|-------|-------|
| T1 segmentation (LoRA enc+dec) | 46,872 | 22,629,850 | **0.207%** |
| T2 regression (LoRA enc) | 28,177 | 14,163,985 | **0.199%** |

The backbone is ~14-23M parameters depending on task head, not ~50M, and the trainable
share is ~0.2%, not <0.1%. Still a strong efficiency claim, but it has to be stated
correctly. `scripts/run_benchmark_v2.py` now records the exact counts in every result file.

---

## 6. The headline comparison is not like-for-like

Under the original baseline definition, `sequential_linear` freezes only the *encoder* and
trains everything downstream. For segmentation that is the entire decoder:

| Method | Trainable (T1 segmentation) | Share |
|--------|-----------------------------|-------|
| LoRA (enc+dec) | 46,872 | 0.21% |
| Sequential Linear | **8,446,306** | 37.4% |
| Sequential FT | 22,582,978 | 100% |

So "LoRA 0.60 Dice vs Sequential Linear 0.79" compares two methods whose trainable budgets
differ by a factor of 180, presented as a straight head-to-head. Either the counts belong
in the table (framing LoRA as a point on a cost/accuracy frontier), or LoRA needs a fair
configuration.

One concrete asymmetry is worth testing on its own: under LoRA the segmentation output
projection `backbone.decoder.out_conv` (2x32x1x1x1 + 2 = **66 parameters**) stays frozen at
its pretrained value, so the model cannot re-map decoder channels onto the new label space
at all — it can only perturb them through rank-8 deltas. The `lora_head` method added to
the driver unfreezes exactly those 66 parameters. If the Dice gap narrows materially, the
reported gap was a configuration artefact rather than a limitation of LoRA, and the paper's
main weakness disappears.

---

## Flagged, needing a decision

**A. Task 1 negatives are degenerate volumes.** All 250 IXI-healthy negatives on disk have
collapsed shapes — `(4, 0, 30, 0)`, `(4, 2, 32, 18)` — while all 250 ISLES positives are
intact (~224³). The current preprocessing code produces correct `(4, 240, 240, 180)`
volumes, so the on-disk data is stale and can be regenerated (~15 min CPU).
Guarded by `test_task1_volumes_are_not_degenerate` (currently failing, by design).

**B. Task 1 has a class–modality confound, and regenerating will not fix it.** ISLES
positives carry real `[DWI, FLAIR, ADC, SWI]`; IXI negatives are `[T1, T2, 0, 0]`. The two
classes differ by *acquisition*, not by pathology, so a classifier can separate them
without learning anything about infarcts — including from the two all-zero channels alone.
Any AUC from this setup would be meaningless. Task 1 should either be built from a cohort
with matched modalities, or stay out of the paper with this stated as the real reason
(the current stated reason — "ISLES lacks healthy negatives" — is not accurate).

---

## What has to be re-run

Everything. No number in `EXPERIMENT_RESULTS.md` §4 or in Tables I–VI of the manuscript
survives unchanged. Staged in `scripts/run_stage.sh`:

GPU work on this machine is scheduled through `gq`; submit with `scripts/submit_stage.sh`
rather than running the drivers directly.

**Measured resource use** (`scripts/probe_vram.py`, gq job #26). The declaration comes from
this, not from a guess:

| Phase | Task | Allocated | Reserved | Wall clock |
|-------|------|-----------|----------|------------|
| train step (batch 2) | segmentation | 1670 MB | 1850 MB | -- |
| sliding-window eval (batch 8) | segmentation | 2359 MB | **4158 MB** | 15.4 s/case |
| train step (batch 2) | regression | 686 MB | 988 MB | -- |
| sliding-window eval (batch 8) | regression | 1371 MB | 2526 MB | 0.6 s/case |

The training figure reproduces the project's own recorded `gpu_max_mb` of 1670.48 exactly,
which is a useful check that the probe measures the right thing. Peak is driven by
*evaluation*, not training — the old protocol never surfaced this because it never ran
whole-volume inference. Declared: 5G for the LoRA/linear methods, 6G for `sequential_ft`
(gradients plus Adam state for all 22.6M parameters add ~270 MB over LoRA's 47K).

**Cost.** Whole-volume scoring is what makes the results comparable to published BraTS
numbers, and it is not cheap: ~64 min per 250-case segmentation test split. Per run that is
roughly 65 min training + 64 min test evaluation for the isolated methods, and ~205 min for
the sequential baselines, which re-score task 1 after task 2 to measure backward transfer.
Stage A totals ~11.5 GPU-hours. If that proves too expensive on a contended machine, the
cheapest lever is `--overlap 0.25` (roughly 3x faster, still a standard setting); reducing
the test set below 250 subjects would widen the confidence intervals instead.

- **Stage A** — `lora`, `lora_head`, `sequential_linear`, `sequential_ft` at seed 42. This
  is the decisive run: it shows whether LoRA still has a claim once regression is real and
  Dice is measured per-case on held-out subjects, and whether the 66-parameter output head
  closes the segmentation gap.
- **Stage B** — seeds 43, 44 for the same four.
- **Stage C** — `lora_encoder_only`, `pertask_lora` oracle.

EWC / LwF / Replay still need porting to the corrected driver before their rows can be
restored.

**Expected outcome to prepare for.** In the old tables Sequential Linear already beat LoRA
on segmentation (0.79 vs 0.60) with near-zero forgetting (BWT −0.01); LoRA's only win was
the regression column, which was an artefact. If that pattern holds under the corrected
protocol, the current headline claim — "LoRA is the only method that maintains balanced
performance" — does not survive, and the paper needs the repositioning discussed
separately (task-ID-free routing, forward transfer, or a benchmark contribution).

## 5. Corrected results — Stage A (clean splits v3, seed 42)

Held-out test set, subject-grouped splits, whole-volume sliding-window inference
(overlap 0.25), per-case metrics. Unlabelled BraTS cases excluded (`hd95_undefined = 0/275`
for every method, confirming the fix). Results in `outputs/benchmark_v2/n32/`.

| Method | Dice↑ | HD95↓ | Age MAE (yr)↓ | Age R² | Dice after T2 | BWT | Trainable |
|---|---|---|---|---|---|---|---|
| LoRA (enc+dec) | 0.452 | 136.5 | 15.04 | −0.122 | = T1 | 0.000 | 0.207% |
| **LoRA + out_conv (`lora_head`)** | **0.794** | **41.5** | 14.58 | −0.038 | = T1 | **0.000** | **0.207%** |
| Sequential Linear | 0.795 | 43.4 | 14.80 | −0.068 | 0.795 | 0.000 | 37.4% |
| Sequential FT | 0.813 | 38.4 | 14.66 | −0.035 | 0.561 | −0.253 | 100% |

**Paired comparison on the 275 identical test cases** (bootstrap 95% CI, Wilcoxon):

| Contrast | Mean ΔDice | 95% CI | p |
|---|---|---|---|
| lora_head − Sequential Linear | −0.002 | [−0.008, +0.005] | 0.28 |
| Sequential FT − lora_head | +0.020 | [+0.014, +0.026] | 7e−16 |
| lora_head − LoRA | **+0.342** | [+0.323, +0.360] | 8e−47 |

### What this establishes

1. **The 66-parameter output projection was the whole segmentation gap.** Unfreezing
   `backbone.decoder.out_conv` lifts Dice by +0.34 (p < 1e−46) at unchanged parameter
   cost. The manuscript's "LoRA underperforms on segmentation" was a configuration
   artefact, not a property of LoRA.
2. **lora_head is statistically indistinguishable from linear probing** (p = 0.28) while
   training 180× fewer parameters (46,938 vs 8,446,306) and beats it on HD95. Only full
   fine-tuning is better, by 2 Dice points — and full fine-tuning forgets 25 points on the
   first task after learning the second.
3. **Brain age does not work for any method at 32 shots.** Every method — including full
   fine-tuning of all 22.6M parameters — lands at MAE 14.6–15.0 yr with negative R² and
   prediction std ≤ 1.2 yr against a true std of 16.5. The constant-predictor baseline
   is 14.61. This is a task-setup limitation (patch-based input, shot count, or backbone
   suitability for regression), not a method result, and must be presented as such.
4. **Zero forgetting is not unique to adapter isolation.** Sequential Linear also has
   BWT = 0 because its encoder is frozen throughout. The paper's differentiator has to be
   parameter cost at equal accuracy, not forgetting alone.

### Single-seed caveat

These are seed 42 only. Stage B (seeds 43, 44) is required before any of the above can be
stated with variance; the paired CIs above are over cases, not over training runs.
