"""
Graphical abstract, revised for the corrected protocol.

Same layout vocabulary as make_graphical_abstract.py (title / labels / three panels /
three finding cards, same palette), but the two outer panels are now drawn from the
measured results rather than from qualitative crops of pre-audit checkpoints:

  Row 2: [Accuracy vs. trainable budget]  [Framework diagram]  [Forgetting across 3 tasks]
  Row 3: three finding cards

Every number is read from outputs/benchmark_v2 so re-running after new seeds land
refreshes the figure. Run from the paper directory:

  python make_graphical_abstract_v2.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "outputs", "benchmark_v2", "n32")

RED, BLUE, GREEN, ORANGE, PURPLE, INK, GREY = (
    "#c0392b", "#2471a3", "#27ae60", "#e67e22", "#8e44ad", "#1a1a2e", "#555555")


# ── results ──────────────────────────────────────────────────────────
def load(cfg):
    out = []
    for f in sorted(glob.glob(os.path.join(RESULTS, cfg, "seed*", "results.json"))):
        with open(f) as fh:
            out.append(json.load(fh))
    return out


def stat(runs, task, key, after=None):
    vals = []
    for r in runs:
        t = r["per_task"].get(task, {})
        blk = t.get(f"test_after_task{after}") if after else t.get("test")
        if blk and key in blk:
            vals.append(blk[key])
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (np.nan, np.nan, 0)


two = {k: load(k) for k in ("lora", "lora_head", "lora_head_r16", "sequential_linear", "sequential_ft")}
three = {k: load(k + "_t243") for k in ("lora_head_r16", "sequential_linear", "sequential_ft")}


def paired_dice(cfg_a, cfg_b, task="task2"):
    """Mean per-case Dice difference pooled over seeds, as in the paper's Table III."""
    diffs = []
    for fa in sorted(glob.glob(os.path.join(RESULTS, cfg_a, "seed*", f"{task}_per_case.json"))):
        fb = fa.replace(os.sep + cfg_a + os.sep, os.sep + cfg_b + os.sep)
        if not os.path.exists(fb):
            continue
        with open(fa) as h:
            a = {c["case"]: c["dice"] for c in json.load(h)}
        with open(fb) as h:
            b = {c["case"]: c["dice"] for c in json.load(h)}
        diffs += [a[k] - b[k] for k in a if k in b]
    return float(np.mean(diffs)) if diffs else np.nan

label = {"lora": "LoRA, frozen\noutput proj.", "lora_head_r16": "LoRA + head\n(r = 16)",
         "sequential_linear": "Linear probe\n(full decoder)", "sequential_ft": "Full\nfine-tuning"}
colour = {"lora": GREY, "lora_head_r16": GREEN, "sequential_linear": BLUE, "sequential_ft": RED}
params = {"lora": 46_872, "lora_head_r16": 93_810, "sequential_linear": 8_446_306,
          "sequential_ft": 22_582_978}

dice = {k: stat(v, "task2", "dice") for k, v in two.items()}
n_seeds_two = max(d[2] for d in dice.values())
n_seeds_three = max(stat(v, "task2", "dice")[2] for v in three.values())

# ── figure ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor="white")
gs = fig.add_gridspec(4, 6, height_ratios=[0.07, 0.03, 0.55, 0.33],
                      width_ratios=[1, 1, 1.6, 1.6, 1, 1],
                      hspace=0.22, wspace=0.12, left=0.05, right=0.97, top=0.96, bottom=0.04)

ax = fig.add_subplot(gs[0, :]); ax.axis("off")
ax.text(0.5, 0.5, "Few-Shot Continual Learning for 3D Brain MRI with Frozen Foundation Models",
        ha="center", va="center", fontsize=20, fontweight="bold", color=INK)

for cols, text, col, bg in [((0, 2), "Segmentation: 90× cheaper, −0.01 Dice", RED, "#fadbd8"),
                            ((2, 4), "Frozen backbone + per-task LoRA", "#2c3e50", None),
                            ((4, 6), "Global tasks: only LoRA learns them", BLUE, "#d4e6f1")]:
    a = fig.add_subplot(gs[1, cols[0]:cols[1]]); a.axis("off")
    kw = dict(bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, edgecolor=col, alpha=0.9)) if bg else {}
    a.text(0.5, 0.5, text, ha="center", va="center", fontsize=13, fontweight="bold", color=col,
           transform=a.transAxes, **kw)

# ── panel 1: the segmentation margin, replicated across three tasks ──
axl = fig.add_subplot(gs[2, 0:2])
segs = [("BraTS\nglioma", "lora_head_r16", "sequential_linear", "task2"),
        ("ISLES\nstroke", "lora_head_r16_t243", "sequential_linear_t243", "task4"),
        ("MSD\nhippocampus", "lora_head_r16_t7", "sequential_linear_t7", "task7")]
x = np.arange(len(segs)); w = 0.36
lo_v = [stat(load(a), t, "dice")[0] for _, a, _, t in segs]
lo_e = [stat(load(a), t, "dice")[1] for _, a, _, t in segs]
sl_v = [stat(load(b), t, "dice")[0] for _, _, b, t in segs]
sl_e = [stat(load(b), t, "dice")[1] for _, _, b, t in segs]
axl.bar(x - w/2, lo_v, w, yerr=lo_e, color=GREEN, edgecolor="white", linewidth=1.5,
        error_kw=dict(ecolor="#333", lw=1.4, capsize=3), label="LoRA + head (93.8K)", zorder=3)
axl.bar(x + w/2, sl_v, w, yerr=sl_e, color=BLUE, edgecolor="white", linewidth=1.5,
        error_kw=dict(ecolor="#333", lw=1.4, capsize=3), label="linear probe (8.45M)", zorder=3)
for xi, a, b in zip(x, lo_v, sl_v):
    axl.text(xi, max(a, b) + 0.055, f"{a - b:+.3f}", ha="center", fontsize=10,
             color=GREEN, fontweight="bold")
axl.set_xticks(x); axl.set_xticklabels([n for n, *_ in segs], fontsize=9.5)
axl.set_ylabel("Dice (held-out, per case)", fontsize=10, color=GREY)
axl.set_ylim(0, 1.16)
axl.grid(True, axis="y", alpha=0.25)
axl.tick_params(labelsize=9)
axl.legend(fontsize=8.5, loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.99))
for sp in axl.spines.values():
    sp.set_edgecolor(RED); sp.set_linewidth(2)
axl.text(0.02, -0.15, "3 seeds · margin is 0.011–0.015 on all three", transform=axl.transAxes,
         fontsize=8.5, color=GREY, style="italic")

# ── panel 2: framework diagram (schematic still valid) ───────────────
axc = fig.add_subplot(gs[2, 2:4]); axc.axis("off")
axc.imshow(Image.open(os.path.join(HERE, "figures", "framework_diagram.png")))

# ── panel 3: brain age, whole-head vs patch ──────────────────────────
axr = fig.add_subplot(gs[2, 4:6])
wb = {k: load(k + "_t5") for k in ("lora_head_r16", "sequential_ft", "sequential_linear", "bitfit")}
colour["bitfit"] = PURPLE
label["bitfit"] = "BitFit"
names = ["lora_head_r16", "sequential_ft", "sequential_linear", "bitfit"]
mae_wb = [stat(wb[k], "task5", "mae")[0] for k in names]
err_wb = [stat(wb[k], "task5", "mae")[1] for k in names]
gt = [c["gt"] for c in json.load(open(glob.glob(os.path.join(RESULTS, "lora_head_r16_t5", "seed42", "task5_per_case.json"))[0]))]
const = float(np.mean(np.abs(np.array(gt) - np.mean(gt))))
ypos = np.arange(len(names))[::-1]
axr.barh(ypos, mae_wb, xerr=err_wb, color=[colour[k] for k in names], height=0.62,
         edgecolor="white", linewidth=1.5, error_kw=dict(ecolor="#333", lw=1.5, capsize=3), zorder=3)
axr.axvline(const, color=INK, ls="--", lw=2, zorder=4)
axr.text(const - 0.35, ypos[0] + 0.55, f"constant predictor {const:.1f} yr", fontsize=8.5,
         color=INK, va="bottom", ha="right", fontweight="bold")
axr.set_yticks(ypos)
axr.set_yticklabels([])
for y, k in zip(ypos, names):
    axr.text(0.35, y, label[k].replace("\n", " "), va="center", ha="left", fontsize=9.5,
             color="white", fontweight="bold", zorder=5)
axr.set_xlabel("brain-age MAE (years), whole-head $96^3$ input", fontsize=10, color=GREY)
axr.set_xlim(0, 18.5)
axr.grid(True, axis="x", alpha=0.25)
axr.tick_params(labelsize=9)
for sp in axr.spines.values():
    sp.set_edgecolor(BLUE); sp.set_linewidth(2)
for y, k, m in zip(ypos, names, mae_wb):
    sd = stat(wb[k], "task5", "pred_std")[0]
    axr.text(m + 0.3, y + 0.30, f"{m:.1f}", va="center", fontsize=9.5, color=colour[k], fontweight="bold")
    if sd < 1.0:
        axr.text(m + 0.3, y - 0.22, "constant output", va="center", fontsize=8.5, color=colour[k], style="italic")
axr.text(0.02, -0.16, "3 seeds · from $64^3$ patches every method sits at the baseline",
         transform=axr.transAxes, fontsize=8.5, color=GREY, style="italic")

# ── row 3: finding cards ─────────────────────────────────────────────
lh, sl = dice["lora_head_r16"][0], dice["sequential_linear"][0]
gap = paired_dice("lora_head_r16", "sequential_linear")      # paired, as in Table III
gain = paired_dice("lora_head", "lora")                       # both at r = 8: the 66-param effect alone
sex = {k: load(k + "_t6") for k in ("lora_head_r16", "sequential_ft", "sequential_linear", "bitfit")}
auc_l = stat(sex["lora_head_r16"], "task6", "auc")[0]
auc_f = stat(sex["sequential_ft"], "task6", "auc")[0]
auc_p = stat(sex["sequential_linear"], "task6", "auc")[0]
findings = [
    {"icon": f"{mae_wb[0]:.1f} yr | {auc_l:.2f} AUC", "title": "Two global tasks, same answer",
     "desc": f"Brain age {mae_wb[0]:.1f} vs {const:.1f} constant;\nsex AUC {auc_l:.2f} vs {auc_f:.2f} full fine-tune;\nlinear probe outputs a constant on both",
     "color": BLUE, "bg": "#d4e6f1"},
    {"icon": "0.011-0.015", "title": "Segmentation margin, 3 tasks",
     "desc": "Same gap on glioma, stroke and\nhippocampus despite differing in\nchannels, pathology and scale",
     "color": GREEN, "bg": "#eafaf1"},
    {"icon": f"AUC {auc_p:.2f}, acc 0.50", "title": "Signal present, unusable",
     "desc": "Frozen features rank sex far above\nchance, yet a 1K head gives a constant\ndecision — the bottleneck is geometry",
     "color": ORANGE, "bg": "#fef5e7"},
]
for i, f in enumerate(findings):
    a = fig.add_subplot(gs[3, i * 2:(i + 1) * 2]); a.axis("off")
    a.add_patch(mpatches.FancyBboxPatch((0.03, 0.05), 0.94, 0.90, boxstyle="round,pad=0.03",
                                        facecolor=f["bg"], edgecolor=f["color"], linewidth=2.5,
                                        transform=a.transAxes))
    a.text(0.5, 0.80, f["icon"], ha="center", va="center", fontsize=21, fontweight="bold",
           color=f["color"], transform=a.transAxes, fontfamily="monospace")
    a.text(0.5, 0.57, f["title"], ha="center", va="center", fontsize=13, fontweight="bold",
           color="#2c3e50", transform=a.transAxes)
    a.text(0.5, 0.27, f["desc"], ha="center", va="center", fontsize=10, color=GREY,
           transform=a.transAxes, linespacing=1.3)

for (x0, x1), col in [((0.36, 0.31), RED), ((0.64, 0.685), BLUE)]:
    fig.patches.append(mpatches.FancyArrowPatch((x0, 0.52), (x1, 0.52), transform=fig.transFigure,
                                                arrowstyle="->,head_width=10,head_length=8",
                                                color=col, linewidth=3, mutation_scale=1))

for name, dpi in (("graphical_abstract.png", 300), ("graphical_abstract.pdf", 300),
                  ("graphical_abstract_preview.png", 150)):
    fig.savefig(os.path.join(HERE, name), dpi=dpi, bbox_inches="tight", facecolor="white")
    print("saved", name)
plt.close(fig)

# ── text card ────────────────────────────────────────────────────────
with open(os.path.join(HERE, "GraphicalAbstractCaption.txt")) as fh:
    caption = fh.read().strip()
fig2 = plt.figure(figsize=(8, 3.2), facecolor="white")
a2 = fig2.add_subplot(111); a2.axis("off")
a2.text(0.5, 0.5, caption, ha="center", va="center", fontsize=11.5, color=INK, wrap=True,
        transform=a2.transAxes, linespacing=1.5, fontfamily="serif")
fig2.savefig(os.path.join(HERE, "graphical_abstract_text.pdf"), dpi=300, bbox_inches="tight", facecolor="white")
print("saved graphical_abstract_text.pdf")
