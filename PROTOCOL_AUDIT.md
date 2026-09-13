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

## 5. Corrected results — three seeds (clean splits v3)

Held-out test set, subject-grouped splits, whole-volume sliding-window inference
(overlap 0.25), per-case metrics. Mean ± SD over seeds 42–44; `hd95_undefined = 0` on every
run. Results in `outputs/benchmark_v2/n32/`.

| Method | Dice↑ | HD95↓ | Age MAE (yr)↓ | Age R² | Dice after T2 | BWT | Trainable |
|---|---|---|---|---|---|---|---|
| LoRA (enc+dec) | 0.538 ± 0.082 | 135.1 ± 14.0 | 14.51 ± 0.39 | −0.06 ± 0.06 | = T1 | 0.000 | 0.207% |
| LoRA + out_conv (`lora_head`) | 0.754 ± 0.029 | 48.6 ± 5.0 | 14.40 ± 0.31 | −0.01 ± 0.02 | = T1 | 0.000 | **0.207%** |
| Sequential Linear | 0.789 ± 0.005 | 41.9 ± 1.3 | 14.64 ± 0.12 | −0.06 ± 0.03 | 0.789 ± 0.005 | 0.000 | 37.4% |
| Sequential FT | **0.798 ± 0.025** | **40.2 ± 4.9** | 14.52 ± 0.10 | −0.03 ± 0.03 | 0.470 ± 0.084 | **−0.328 ± 0.062** | 100% |

**Paired per-case Dice, pooled over seeds** (n = 820; bootstrap 95% CI; Wilcoxon):

| Contrast | Mean ΔDice | 95% CI | p | per-seed |
|---|---|---|---|---|
| lora_head − Sequential Linear | **−0.035** | [−0.040, −0.030] | 9e−50 | −0.002 (p=0.28), −0.061, −0.043 |
| Sequential FT − lora_head | +0.044 | [+0.040, +0.049] | 8e−82 | |
| lora_head − LoRA | **+0.217** | [+0.207, +0.228] | 1e−134 | +0.342, +0.213, +0.094 |

### What this establishes — and what the seed-42 draft got wrong

1. **The 66-parameter output projection is most of the LoRA segmentation gap, not all of
   it.** +0.217 Dice pooled (+0.34 / +0.21 / +0.09 by seed). Plain LoRA's spread
   (0.452–0.648) is three times wider than lora_head's, so the frozen projection also
   makes the outcome seed-dependent.
2. **lora_head does *not* match linear probing.** The seed-42 result (Δ = −0.002, p = 0.28)
   was the outlier; seeds 43 and 44 give −0.061 and −0.043. Pooled: −0.035 Dice, 4.4%
   relative, p ≈ 1e−50. Linear probing is also better on HD95 (41.9 vs 48.6) and far more
   stable (SD 0.005 vs 0.029). The earlier "statistically indistinguishable" and "beats it
   on HD95" claims are withdrawn from the manuscript.
3. **The honest claim is a trade-off:** −0.035 Dice for 180× fewer trainable parameters
   (46.9K vs 8.45M) and ~170× less per-task storage (<200 KB vs ~34 MB). That is a
   legitimate Pareto point, not a dominance result.
4. **Full fine-tuning forgets more than seed 42 suggested:** BWT −0.328 ± 0.062
   (−0.25 / −0.33 / −0.40), which wipes out its +0.044 accuracy edge several times over.
5. **Zero forgetting is shared with Sequential Linear on every seed**, and Sequential Linear
   is the more accurate of the two forgetting-free methods.
6. **Brain age is a negative result for every method on every seed** (MAE 14.4–14.6 vs
   constant ≈14.6; R² within ±0.06 of zero; prediction SD ≤ 2.7 vs true 16.5). Method
   ordering changes from seed to seed, as it would for noise.

**Lesson recorded for the paper:** a single seed would have supported a claim the data does
not. The manuscript now reports three seeds throughout and says so in Limitations.

## 6. Rank ablation — LoRA + head, r ∈ {4, 8, 16} (three seeds)

Same protocol and splits as §5. `outputs/benchmark_v2/n32/lora_head{,_r4,_r16}/`.

| Config | Dice↑ | HD95↓ | Age MAE | Trainable (T1) | vs. Seq. Linear (paired, n=820) |
|---|---|---|---|---|---|
| LoRA + head, r=4 | 0.761 ± 0.023 | 50.1 ± 3.3 | 14.35 | 23,502 (0.104%) | — |
| LoRA + head, r=8 | 0.754 ± 0.029 | 48.6 ± 5.0 | 14.40 | 46,938 (0.207%) | −0.035 [−0.040, −0.030], p=9e−50 |
| **LoRA + head, r=16** | **0.777 ± 0.004** | **43.2 ± 0.7** | 14.72 | 93,810 (0.414%) | **−0.012 [−0.015, −0.008], p=3e−10** |
| Sequential Linear | 0.789 ± 0.005 | 41.9 ± 1.3 | 14.64 | 8,446,306 (37.4%) | reference |
| Sequential FT | 0.798 ± 0.025 | 40.2 ± 4.9 | 14.52 | 22,582,978 (100%) | +0.009 over SL; **+0.021 over r=16** (p=1e−33) |

Per-seed r=16 − Seq. Linear: −0.013, −0.013, −0.009 — the deficit is small and the same
on every seed, unlike r=8 where it ranged from −0.002 to −0.061.

**What changes:** the honest headline moves from "−0.035 Dice for 180× fewer parameters"
to "**−0.012 Dice (1.5% relative) for 90× fewer parameters, with seed stability matching
linear probing** (SD 0.004 vs 0.005)". r=16 is also more stable than r=8 by a factor of
seven. The gap to full fine-tuning is 0.021, and full fine-tuning forgets 0.33.

r=4 is not worse than r=8 (+0.007, p=9e−6) at half the parameters — rank 8 sits in a flat
region; rank 16 is where the extra capacity starts to pay. Brain age is unchanged by rank
(negative result holds).

**Manuscript action:** make r=16 the reported configuration in Table II, keep r=4/8/16 as an
ablation table, and rewrite the trade-off sentences in the abstract, contributions, §V-C,
Discussion and Conclusion to the r=16 numbers. Pending the ISLES pilot, which may
restructure §V.

## 7. JBHI must-list — status (2026-09-13)

| Item | Status | Where |
|---|---|---|
| Rank ablation r ∈ {4, 8, 16} | **done**, 3 seeds — r=16 reported (§6) | `lora_head{,_r4,_r16}/` |
| Third task that works (ISLES 2022 lesion seg) | data prepared (247 cases); **pilot queued** #820–822, sequence 2→4→3, seed 42 | `*_t243/` |
| Shot ablation 16 / 32 / 64 | **queued** #823–834: lora_head r16 + Seq. Linear × 3 seeds | `n16/`, `n64/` |
| Random-init backbone control | **queued** #835–837: lora_head r16, no pretrained weights | `lora_head_r16_randinit/` |
| EWC / LwF / Replay under the corrected protocol | ported, smoke-tested; **queued** #841–849, 3 seeds each, declared 6G | `ewc/`, `lwf/`, `replay/` |
| Brain age with whole-brain input | 559 cubes at 96³, splits built, smoke-tested; **queued** #850–861: lora_head r16 / Seq. Linear / Seq. FT / BitFit × 3 seeds, `--tasks 5 --patch_size 96`, declared 5G from ×3.4 voxel scaling of the 64³ measurement | `*_t5/` |
| PEFT comparator: BitFit | **queued** #838–840 (bias + output projection, 6.4K params = 0.028%) | `bitfit/` |
| PEFT comparator: Adapter (Houlsby) | not planned — needs module insertion into the 3D U-Net; BitFit covers the low-budget comparison | — |
| Extra seeds (45, 46) | not started | — |

Queue depth is the constraint: 18 jobs of mine behind other users' at 8/8 slots. Runs are
declared at 5G/6G VRAM, 2 CPU, 4G RAM from measurement.
