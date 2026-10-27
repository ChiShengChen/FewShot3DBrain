"""
Graphical abstract v3: the nine tasks as real slices, the decision rule, three findings.

  Row 1  one held-out test case per task (seed-42 split), segmentation masks overlaid,
         grouped into "large task head" (segmentation) and "small head" (global) families
  Row 2  the rule: which strategy each family calls for, with the measured margins
  Row 3  three finding cards (forgetting, pretraining, protocol), numbers read from
         outputs/benchmark_v2 and results/stats_v2.json so the figure follows the data

Run from the paper directory:  python make_graphical_abstract_v3.py
"""
import glob
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PRE = os.path.join(ROOT, "data", "preprocessed")
RES = os.path.join(ROOT, "outputs", "benchmark_v2", "n32")

BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
INK, INK2, MUTED, LINE, PANEL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#f4f4f1"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})


# ── data helpers ───────────────────────────────────────────────────────────────
def test_case(task_name, i=0):
    sp = json.load(open(os.path.join(ROOT, "data/splits_v3", f"{task_name}_n32_seed42.json")))["splits"]["test"]
    base = sp[i]
    return base if os.path.isabs(base) else os.path.join(ROOT, base)


CUBE_AXIS = int(os.environ.get("CUBE_AXIS", "1"))   # axis 1 of the 96^3 cubes is axial


def square(a, fill=0.0):
    h, w = a.shape[:2]
    n = max(h, w)
    out = np.full((n, n) + a.shape[2:], fill, dtype=a.dtype)
    y, x = (n - h) // 2, (n - w) // 2
    out[y:y + h, x:x + w] = a
    return out


def slice_of(base, channel=0, with_mask=True):
    v = np.load(base + ".npy", mmap_mode="r")
    dims = v.shape[1:]
    ax = CUBE_AXIS if len(set(dims)) == 1 else int(np.argmin(dims))   # thinnest axis = slice direction
    if with_mask:
        s = np.load(base + "_seg.npy", mmap_mode="r")[0]
        area = s.sum(axis=tuple(a for a in range(3) if a != ax))
        k = int(area.argmax())
        m = np.take(s, k, axis=ax)
    else:
        k = v.shape[1 + ax] // 2
        m = None
    img = np.take(v[channel], k, axis=ax).astype(np.float32)
    fg = img != 0
    lo, hi = np.percentile(img[fg], [1, 99]) if fg.any() else (0, 1)
    img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    img[~fg] = 0.0                                          # z-scored background is exactly 0: keep it black
    img = square(np.rot90(img))
    return img, (square(np.rot90(m)) if m is not None else None)


def runs(cfg):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, cfg, "seed*", "results.json")))]


def mean(cfg, task, key, stage="test"):
    v = [r["per_task"][f"task{task}"][stage][key] for r in runs(cfg) if stage in r["per_task"].get(f"task{task}", {})]
    return float(np.mean(v)) if v else np.nan


stats = json.load(open(os.path.join(ROOT, "results", "stats_v2.json")))
def contrast(fam, prefix):
    for r in stats[fam]:
        if r["label"].startswith(prefix):
            return r
    return None


# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9.6), facecolor="white")
G = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[2.9, 1.7, 2.0], hspace=0.12, left=0.03, right=0.97, top=0.875, bottom=0.03)

fig.text(0.03, 0.965, "Which adaptation strategy for a new brain-MRI task?", fontsize=20, weight="bold", color=INK, va="center")
fig.text(0.03, 0.932, "One rule decides it, and it is knowable before training: does the new task inherit a large task-specific head?  "
         "Nine tasks, six cohorts, 32 labelled volumes each, frozen FOMO 3D U-Net, five seeds, 301 runs.",
         fontsize=11, color=INK2, va="center")

# Row 1: brains ---------------------------------------------------------------------
cases = [
    # (family, title, task_name, channel, mask, label)
    ("seg", "BraTS 2023 glioma", "Task002_FOMO2", 1, True, "3 ch · 1,251 vol."),
    ("seg", "ISLES 2022 stroke", "Task004_ISLES_SEG", 0, True, "DWI/ADC/FLAIR · 247"),
    ("seg", "MSD hippocampus", "Task007_HIPPO", 0, True, "T1 · 260"),
    ("seg", "WMH 2017", "Task008_WMH_SEG", 0, True, "FLAIR+T1 · 170 · 5 scanners"),
    ("glob", "IXI brain age", "Task005_IXI_WB", 0, False, None),
    ("glob", "IXI sex", "Task006_IXI_SEX", 0, False, None),
    ("glob", "OASIS-1 brain age", "Task009_OASIS_AGE", 0, False, None),
    ("glob", "OASIS-1 sex", "Task010_OASIS_SEX", 0, False, None),
    ("glob", "OASIS-1 dementia (CDR>0)", "Task011_OASIS_CDR", 0, False, None),
]
row = G[0].subgridspec(1, 9, wspace=0.06)
for i, (fam, title, tname, ch, msk, sub) in enumerate(cases):
    ax = fig.add_subplot(row[0, i])
    base = test_case(tname, 0)
    img, m = slice_of(base, ch, msk)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
    if m is not None and m.any():
        over = np.zeros(img.shape + (4,)); over[m > 0] = matplotlib.colors.to_rgba(RED, 0.55)
        ax.imshow(over, interpolation="nearest")
    if not msk:
        lab = open(base + ".txt").read().strip()
        if "SEX" in tname:
            lab = "female" if lab.strip() == "1" else "male"
        elif "CDR" in tname:
            lab = "CDR > 0" if lab.strip() == "1" else "CDR 0"
        else:
            lab = f"{float(lab):.0f} y"
        ax.text(0.5, 0.04, f"label: {lab}", transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc=INK, ec="none", alpha=0.75))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(ORANGE if fam == "seg" else BLUE); sp.set_linewidth(2.2)
    ax.set_title(title, fontsize=9.5, color=INK, pad=4)
    if sub:
        ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=MUTED)
    if i == 0:
        fig.text(ax.get_position().x0, ax.get_position().y1 + 0.045, "LARGE TASK HEAD  ·  segmentation inherits an 8.45M-parameter decoder",
                 fontsize=10.5, weight="bold", color=ORANGE, va="bottom")
    if i == 4:
        fig.text(ax.get_position().x0, ax.get_position().y1 + 0.045, "SMALL HEAD  ·  a ~1K-parameter regressor or classifier over pooled features",
                 fontsize=10.5, weight="bold", color=BLUE, va="bottom")

# Row 2: the rule --------------------------------------------------------------------
r2 = G[1].subgridspec(1, 2, wspace=0.05)
seg_margins = [contrast("seg", p)["mean"] for p in ("BraTS: L+h$_{16}$ $-$ SL", "Hippocampus: SL", "ISLES: SL", "WMH: SL")]
seg_txt = (f"Train the decoder on the frozen encoder. LoRA + head stays within {max(abs(x) for x in seg_margins):.3f} Dice of it on all four\n"
           f"tasks (BraTS {seg_margins[0]:+.3f}, ISLES {-seg_margins[2]:+.3f}, hippocampus {-seg_margins[1]:+.3f}, WMH {-seg_margins[3]:+.3f}) while training 90× fewer\n"
           f"parameters and storing 366 KB per task instead of 32 MB. Pick LoRA when storage matters, the decoder when it does not;\n"
           f"full fine-tuning is more accurate still if the model will only ever serve one task (WMH +{contrast('seg', 'WMH: FT')['mean']:.3f}).")
ixi_mae, oas_mae = mean("lora_head_r16_t5", 5, "mae"), mean("lora_head_r16_t9", 9, "mae")
ixi_auc, oas_auc = mean("lora_head_r16_t6", 6, "auc"), mean("lora_head_r16_t10", 10, "auc")
sl_mae_i, sl_mae_o = mean("sequential_linear_t5", 5, "mae"), mean("sequential_linear_t9", 9, "mae")
glob_txt = (f"Use LoRA: it is the only frozen-backbone strategy that learns these tasks at all. Brain age MAE {ixi_mae:.1f} y (IXI) and\n"
            f"{oas_mae:.1f} y (OASIS) against constant baselines of {sl_mae_i:.1f} and {sl_mae_o:.1f}; sex AUC {ixi_auc:.2f} and {oas_auc:.2f}. Linear probing and\n"
            f"bias-only tuning return a constant output on every one of these tasks, even though their frozen features rank subjects\n"
            f"far above chance: the signal is present, but a small head cannot move the representation to use it.")
for j, (col, head, body) in enumerate(((ORANGE, "→  Train the existing head (or LoRA, for the storage)", seg_txt),
                                       (BLUE, "→  LoRA is necessary, not merely economical", glob_txt))):
    ax = fig.add_subplot(r2[0, j]); ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.01,rounding_size=0.02",
                                         fc=PANEL, ec=col, lw=2, transform=ax.transAxes))
    ax.text(0.04, 0.84, head, transform=ax.transAxes, fontsize=12.5, weight="bold", color=col, va="center")
    ax.text(0.04, 0.66, textwrap.fill(" ".join(body.split()), 104), transform=ax.transAxes, fontsize=8.6, color=INK, va="top", linespacing=1.45)

# Row 3: finding cards ---------------------------------------------------------------
r3 = G[2].subgridspec(1, 3, wspace=0.05)
ft_b = mean("sequential_ft_t243", 2, "dice"); ft_b_after = mean("sequential_ft_t243", 2, "dice", "test_after_task4")
ft_i = mean("sequential_ft_t342", 4, "dice"); ft_i_after = mean("sequential_ft_t342", 4, "dice", "test_after_task2")
bwt = {k: mean(k, 2, "dice", "test_after_task3") - mean(k, 2, "dice") for k in ("sequential_ft", "ewc", "lwf", "replay")}
ri = {k: contrast("seg", k)["mean"] for k in ("BraTS: L+h$_{16}$ $-$ RI", "ISLES: L+h$_{16}$ $-$ RI", "WMH: L+h$_{16}$ $-$ RI")}
cards = [
    (AQUA, "Only freezing prevents forgetting, in either order",
     f"Full fine-tuning: a second segmentation task erases the first whichever comes first\n"
     f"— BraTS Dice {ft_b:.2f} → {ft_b_after:.2f} after ISLES, and ISLES {ft_i:.2f} → {ft_i_after:.2f} after BraTS.\n"
     f"EWC forgets as much as plain fine-tuning (BWT {bwt['ewc']:+.2f} vs {bwt['sequential_ft']:+.2f}); LwF and replay\n"
     f"forget more ({bwt['lwf']:+.2f}, {bwt['replay']:+.2f}). LoRA and linear probing: BWT = 0 by construction."),
    (YELLOW, "Pretraining is a small, task-dependent effect",
     f"Same LoRA configuration on a randomly initialised backbone: identical on BraTS\n"
     f"({ri['BraTS: L+h$_{16}$ $-$ RI']:+.3f} Dice) and IXI brain age, better without pretraining on ISLES ({ri['ISLES: L+h$_{16}$ $-$ RI']:+.3f}),\n"
     f"and a modest gain elsewhere (WMH {ri['WMH: L+h$_{16}$ $-$ RI']:+.3f}; sex +0.03 to +0.11 accuracy).\n"
     f"Validate a checkpoint with a random-init control; do not assume transfer."),
    (INK2, "A protocol that cannot flatter a method",
     "Subject-grouped train / val / test splits, whole-volume sliding-window inference,\n"
     "per-case Dice and HD95, five seeds, 301 runs, paired per-case tests with Holm\n"
     "correction. The pipeline we started with scored random crops on the selection\n"
     "split and inverted the ranking of methods. Code, splits and predictions released."),
]
for j, (col, head, body) in enumerate(cards):
    ax = fig.add_subplot(r3[0, j]); ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.01,rounding_size=0.02",
                                         fc="white", ec=LINE, lw=1.2, transform=ax.transAxes))
    ax.add_patch(mpatches.Rectangle((0.01, 0.02), 0.012, 0.96, fc=col, ec="none", transform=ax.transAxes))
    ax.text(0.06, 0.86, head, transform=ax.transAxes, fontsize=11.5, weight="bold", color=INK, va="center")
    ax.text(0.06, 0.70, textwrap.fill(" ".join(body.split()), 74), transform=ax.transAxes, fontsize=8.3, color=INK, va="top", linespacing=1.5)

for name, dpi in (("graphical_abstract_v3.png", 300), ("graphical_abstract_v3.pdf", 300), ("graphical_abstract_v3_preview.png", 110)):
    fig.savefig(os.path.join(HERE, name), dpi=dpi, bbox_inches="tight", facecolor="white")
print("written graphical_abstract_v3.{png,pdf}")
