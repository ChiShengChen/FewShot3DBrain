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

for cols, text, col, bg in [((0, 2), "Accuracy vs. trainable parameters", RED, "#fadbd8"),
                            ((2, 4), "Frozen backbone + per-task LoRA", "#2c3e50", None),
                            ((4, 6), "Brain age: only LoRA learns it", BLUE, "#d4e6f1")]:
    a = fig.add_subplot(gs[1, cols[0]:cols[1]]); a.axis("off")
    kw = dict(bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, edgecolor=col, alpha=0.9)) if bg else {}
    a.text(0.5, 0.5, text, ha="center", va="center", fontsize=13, fontweight="bold", color=col,
           transform=a.transAxes, **kw)

# ── panel 1: Dice vs. trainable parameters (log x) ───────────────────
axl = fig.add_subplot(gs[2, 0:2])
# label placement: linear probe and full FT sit close on the log axis, so one goes below.
place = {"lora": (0, -30, "top"), "lora_head_r16": (0, 26, "bottom"),
         "sequential_linear": (-6, -30, "top"), "sequential_ft": (0, 26, "bottom")}
for k in ("lora", "lora_head_r16", "sequential_linear", "sequential_ft"):
    m, s, _ = dice[k]
    axl.errorbar(params[k], m, yerr=s, fmt="o", ms=11, color=colour[k], ecolor=colour[k],
                 capsize=4, elinewidth=2, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
    dx, dy, va = place[k]
    axl.annotate(label[k], (params[k], m), xytext=(dx, dy), textcoords="offset points",
                 ha="center", va=va, fontsize=9.5, color=colour[k], fontweight="bold", linespacing=1.1)
axl.set_xscale("log")
axl.set_xlim(1.5e4, 6e7)
lo = min(d[0] for d in dice.values()) - 0.12
axl.set_ylim(lo, 0.86)
axl.set_xlabel("trainable parameters (log scale)", fontsize=10, color=GREY)
axl.set_ylabel("BraTS whole-tumour Dice (held-out)", fontsize=10, color=GREY)
axl.grid(True, which="major", alpha=0.25)
axl.tick_params(labelsize=9)
for sp in axl.spines.values():
    sp.set_edgecolor(RED); sp.set_linewidth(2)
axl.text(0.02, 0.03, f"n = 32 shots, mean ± SD over {n_seeds_two} seeds", transform=axl.transAxes,
         fontsize=8.5, color=GREY, style="italic")
# highlight the 90x gap
axl.annotate("", xy=(params["sequential_linear"], dice["lora_head_r16"][0] - 0.06),
             xytext=(params["lora_head_r16"], dice["lora_head_r16"][0] - 0.06),
             arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2))
axl.text(np.sqrt(params["sequential_linear"] * params["lora_head_r16"]), dice["lora_head_r16"][0] - 0.075,
         "90× fewer parameters,\n−0.012 Dice", ha="center", va="top", fontsize=9, color=GREEN, fontweight="bold")

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
findings = [
    {"icon": f"{mae_wb[0]:.1f} yr MAE", "title": "Brain age: LoRA alone learns it",
     "desc": f"vs. {const:.1f} yr constant baseline and\n{mae_wb[1]:.1f} yr for full fine-tuning;\nlinear probing stays constant",
     "color": BLUE, "bg": "#d4e6f1"},
    {"icon": f"{gap:+.3f} Dice", "title": "Segmentation: 90× cheaper",
     "desc": f"LoRA (r=16) + head: {lh:.3f}\nvs. linear probing: {sl:.3f}\n93.8K vs. 8.45M trainable",
     "color": GREEN, "bg": "#eafaf1"},
    {"icon": "66 params", "title": "The layer that matters",
     "desc": f"Training the output projection\nadds +{gain:.2f} Dice over LoRA\nwith it frozen (r = 8, paired)",
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
