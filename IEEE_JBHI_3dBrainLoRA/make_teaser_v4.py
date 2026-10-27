"""
Teaser / graphical abstract in the style of a conference Figure 1:
  (a) the nine tasks as real slices, grouped by task-head size
  (b) the adaptation strategies as a schematic of what is frozen (snowflake) and trained (flame)
  (c) two small result panels that carry the decision rule
Numbers are read from outputs/benchmark_v2 and results/stats_v2.json.

  python make_teaser_v4.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "outputs", "benchmark_v2", "n32")

# palette: frozen = cool grey-blue, trainable = warm orange; series as in the paper's figures
FROZ, FROZ_E = "#e9eef6", "#8aa0c0"
TRAIN, TRAIN_E = "#fde8d8", "#e8763a"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
INK, INK2, MUTED, LINE = "#111111", "#444444", "#8a8a8a", "#d9d9d9"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8})


# ── data ──────────────────────────────────────────────────────────────────────
def test_case(task_name, i=0):
    b = json.load(open(os.path.join(ROOT, "data/splits_v3", f"{task_name}_n32_seed42.json")))["splits"]["test"][i]
    return b if os.path.isabs(b) else os.path.join(ROOT, b)


def square(a, fill=0.0):
    h, w = a.shape[:2]; n = max(h, w)
    out = np.full((n, n) + a.shape[2:], fill, dtype=a.dtype)
    out[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = a
    return out


def slice_of(base, channel=0, with_mask=True, cube_axis=1):
    v = np.load(base + ".npy", mmap_mode="r"); dims = v.shape[1:]
    ax = cube_axis if len(set(dims)) == 1 else int(np.argmin(dims))
    if with_mask:
        s = np.load(base + "_seg.npy", mmap_mode="r")[0]
        k = int(s.sum(axis=tuple(a for a in range(3) if a != ax)).argmax()); m = np.take(s, k, axis=ax)
    else:
        k = dims[ax] // 2; m = None
    img = np.take(v[channel], k, axis=ax).astype(np.float32); fg = img != 0
    lo, hi = np.percentile(img[fg], [1, 99]) if fg.any() else (0, 1)
    img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1); img[~fg] = 0
    return square(np.rot90(img)), (square(np.rot90(m)) if m is not None else None)


def runs(cfg):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, cfg, "seed*", "results.json")))]


def ms(cfg, task, key, stage="test"):
    v = [r["per_task"][f"task{task}"][stage][key] for r in runs(cfg) if stage in r["per_task"].get(f"task{task}", {})]
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)


# ── drawing helpers ───────────────────────────────────────────────────────────
def box(ax, x, y, w, h, kind, text="", fs=8, lw=1.0, ls="-", weight="normal", radius=0.012):
    fc, ec = {"froz": (FROZ, FROZ_E), "train": (TRAIN, TRAIN_E), "plain": ("white", "#999999"), "none": ("none", "#999999")}[kind]
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=fc, ec=ec, lw=lw, ls=ls, clip_on=False))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight=weight, linespacing=1.15, clip_on=False)


def arrow(ax, x0, y0, x1, y1, lw=0.9, color=INK2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0, mutation_scale=9))


def flame(ax, x, y, s=0.016):
    pts = np.array([[0, -1], [0.55, -0.6], [0.45, 0.1], [0.75, 0.35], [0.25, 1.0], [-0.1, 0.45], [-0.45, 0.9], [-0.7, 0.2], [-0.5, -0.6]]) * s
    ax.add_patch(mp.Polygon(pts + [x, y], closed=True, fc="#f0652b", ec="#c74a12", lw=0.5, zorder=5, clip_on=False))
    ax.add_patch(mp.Polygon(pts * 0.5 + [x, y - 0.3 * s], closed=True, fc="#ffc83d", ec="none", zorder=6, clip_on=False))


def snow(ax, x, y, fs=9):
    ax.text(x, y, "❄", ha="center", va="center", fontsize=fs, color="#4a78b8", zorder=6, clip_on=False)


# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 6.4), facecolor="white")
gs = fig.add_gridspec(1, 3, width_ratios=[4.2, 4.6, 4.2], wspace=0.08, left=0.015, right=0.99, top=0.88, bottom=0.15)

# (a) tasks -------------------------------------------------------------------
A = fig.add_subplot(gs[0]); A.set_xlim(0, 1); A.set_ylim(0, 1); A.axis("off")
A.text(0, 1.0, "(a) Nine tasks, six cohorts, 32 labelled volumes each", fontsize=10.5, weight="bold", color=INK, va="bottom")
seg = [("BraTS glioma", "Task002_FOMO2", 1), ("ISLES stroke", "Task004_ISLES_SEG", 0), ("Hippocampus", "Task007_HIPPO", 0), ("WMH, 5 scanners", "Task008_WMH_SEG", 0)]
glob_ = [("IXI age", "Task005_IXI_WB"), ("IXI sex", "Task006_IXI_SEX"), ("OASIS age", "Task009_OASIS_AGE"), ("OASIS sex", "Task010_OASIS_SEX"), ("OASIS CDR", "Task011_OASIS_CDR")]
gap = 0.02
def tiles(items, y0, with_mask, color, head):
    x0 = 0.03
    w = (0.94 - (len(items) - 1) * gap) / len(items)
    A.add_patch(mp.FancyBboxPatch((0.005, y0 - 0.07), 0.99, w + 0.145, boxstyle="round,pad=0,rounding_size=0.015", fc="none", ec=color, lw=1.1, ls=(0, (4, 2.5))))
    A.text(0.02, y0 + w + 0.025, head, fontsize=8.6, weight="bold", color=color, va="bottom")
    for i, it in enumerate(items):
        name, tname = it[0], it[1]
        x = x0 + i * (w + gap)
        img, m = slice_of(test_case(tname), it[2] if with_mask else 0, with_mask)
        ext = (x, x + w, y0, y0 + w)
        A.imshow(img, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="bilinear", zorder=2)
        if m is not None:
            ov = np.zeros(img.shape + (4,)); ov[m > 0] = matplotlib.colors.to_rgba(RED, 0.6)
            A.imshow(ov, extent=ext, interpolation="nearest", zorder=3)
        A.add_patch(mp.Rectangle((x, y0), w, w, fc="none", ec=color, lw=0.8, zorder=4))
        A.text(x + w / 2, y0 - 0.012, name, ha="center", va="top", fontsize=7.4, color=INK2)
    return
tiles(seg, 0.57, True, ORANGE, "Large task head: inherits the 8.45M-parameter decoder")
tiles(glob_, 0.12, False, BLUE, "Small task head: ~1K-parameter head on pooled features")
A.set_aspect("equal", adjustable="box")

# (b) strategies ----------------------------------------------------------------
B = fig.add_subplot(gs[1]); B.set_xlim(0, 1); B.set_ylim(0, 1); B.axis("off")
B.text(0, 1.0, "(b) Adaptation strategies: what is frozen and what is trained", fontsize=10.5, weight="bold", color=INK, va="bottom")
# backbone schematic, one row per strategy
rows = [("LoRA + head", "r=16, 94K trainable", "lora"), ("Linear probe", "decoder / head, 8.45M | 1K", "probe"),
        ("BitFit", "biases + head, 6K", "bitfit"), ("Full fine-tuning", "22.6M, forgets", "ft")]
y_top, rh = 0.84, 0.21
XE, XD, XH = 0.35, 0.645, 0.86          # left edge of encoder, decoder, head columns
BW, BG, BH = 0.062, 0.012, 0.11         # block width, gap, height
for cx, lab in ((XE + 1.5 * BW + BG, "encoder"), (XD + BW + BG / 2, "decoder"), (XH + 0.04, "head")):
    B.text(cx, y_top + 0.12, lab, ha="center", fontsize=8, color=MUTED)
for r, (label, sub, kind) in enumerate(rows):
    y = y_top - r * rh
    B.text(0.0, y + 0.045, label, fontsize=9, color=INK, weight="bold", va="center")
    B.text(0.0, y + 0.005, sub, fontsize=7.2, color=INK2, va="center")
    enc_train, dec_train = kind == "ft", kind in ("probe", "ft")
    for col_x, n, trained in ((XE, 3, enc_train), (XD, 2, dec_train)):
        for i in range(n):
            x = col_x + i * (BW + BG)
            box(B, x, y - 0.03, BW, BH, "train" if trained else "froz", lw=0.8)
            if kind == "lora":      # low-rank branch drawn beside each block
                box(B, x + 0.006, y - 0.058, BW - 0.012, 0.022, "train", lw=0.6, radius=0.004)
            if kind == "bitfit":    # bias slot in the corner of each block
                box(B, x + BW - 0.022, y + BH - 0.055, 0.016, 0.02, "train", lw=0.5, radius=0.004)
        (flame if trained or kind in ("lora", "bitfit") else snow)(B, col_x - 0.028, y + 0.025)
    box(B, XH, y - 0.012, 0.08, 0.075, "train", "head", fs=7, lw=0.8); flame(B, XH - 0.02, y + 0.06, 0.012)
    arrow(B, XE + 3 * BW + 2 * BG + 0.005, y + 0.025, XD - 0.035, y + 0.025)
    arrow(B, XD + 2 * BW + BG + 0.005, y + 0.025, XH - 0.03, y + 0.025)
    arrow(B, XH + 0.085, y + 0.025, 0.97, y + 0.025); B.text(0.985, y + 0.025, "ŷ", fontsize=9, va="center", color=INK)
# legend
box(B, 0.0, -0.07, 0.04, 0.04, "froz"); snow(B, 0.02, -0.05, 8); B.text(0.05, -0.05, "frozen", fontsize=8, va="center", color=INK2)
box(B, 0.16, -0.07, 0.04, 0.04, "train"); flame(B, 0.18, -0.05, 0.013); B.text(0.21, -0.05, "trained for the new task", fontsize=8, va="center", color=INK2)
B.text(0.0, -0.105, "Backbone: FOMO 3D U-Net.  32 labelled volumes, 100 epochs, identical recipe for every row; one adapter / head per task.", fontsize=7.2, va="center", color=MUTED)

# (c) results -------------------------------------------------------------------
C = gs[2].subgridspec(2, 1, hspace=0.55)
methods = [("LoRA+head", BLUE), ("Linear probe", ORANGE), ("BitFit", YELLOW), ("Full FT", AQUA)]
cfg = {"LoRA+head": "lora_head_r16", "Linear probe": "sequential_linear", "BitFit": "bitfit", "Full FT": "sequential_ft"}
def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(LINE); ax.spines["bottom"].set_color(LINE); ax.tick_params(colors=INK2, length=0, labelsize=7.5)
    ax.grid(axis="y", color="#eeeeee", lw=0.7); ax.set_axisbelow(True)
# top: segmentation Dice
ax = fig.add_subplot(C[0]); style(ax)
tasks = [("BraTS", "", 2), ("ISLES", "_t4", 4), ("Hippo.", "_t7", 7), ("WMH", "_t8", 8)]
w = 0.19
for mi, (mname, col) in enumerate(methods):
    xs, ys, es = [], [], []
    for ti, (tl, suf, task) in enumerate(tasks):
        c = cfg[mname] + suf
        if task == 4 and mname in ("Linear probe", "Full FT"): c = cfg[mname] + "_t243"
        m, s = ms(c, task, "dice"); xs.append(ti + (mi - 1.5) * w); ys.append(m); es.append(s)
    ax.bar(xs, ys, w - 0.03, color=col, label=mname, zorder=2); ax.errorbar(xs, ys, es, fmt="none", ecolor=INK2, elinewidth=0.6, zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels([t[0] for t in tasks]); ax.set_ylim(0, 1.28); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel("Dice", fontsize=8)
ax.set_title("(c) Large head → train it: LoRA within 0.015 Dice of the probe; BitFit collapses", fontsize=9, weight="bold", loc="left", color=INK, pad=6)
ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, 1.02), handlelength=1.0, columnspacing=0.9)
# bottom: global tasks as "fraction of the gap to the constant baseline closed" -> simpler: AUC and R2-based score
ax = fig.add_subplot(C[1]); style(ax)
gtasks = [("IXI age", 5, "_t5", "r2"), ("OASIS age", 9, "_t9", "r2"), ("IXI sex", 6, "_t6", "auc"), ("OASIS sex", 10, "_t10", "auc")]
for mi, (mname, col) in enumerate(methods):
    xs, ys, es = [], [], []
    for ti, (tl, task, suf, key) in enumerate(gtasks):
        m, s = ms(cfg[mname] + suf, task, key); xs.append(ti + (mi - 1.5) * w); ys.append(max(m, 0) if key == "r2" else m); es.append(s)
    ax.bar(xs, ys, w - 0.03, color=col, zorder=2); ax.errorbar(xs, ys, es, fmt="none", ecolor=INK2, elinewidth=0.6, zorder=3)
ax.axhline(0.5, xmin=0.5, xmax=1.0, color=MUTED, lw=0.7, ls=(0, (3, 3))); ax.text(3.45, 0.52, "chance", fontsize=6.5, color=MUTED, ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels([t[0] for t in gtasks]); ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel("R$^2$  |  AUC", fontsize=8)
ax.set_title("Small head → only LoRA learns: probe and BitFit output a constant", fontsize=9, weight="bold", loc="left", color=INK, pad=6)

# bottom strip: three takeaways -------------------------------------------------
ft_b, _ = ms("sequential_ft_t243", 2, "dice"); ft_b2, _ = ms("sequential_ft_t243", 2, "dice", "test_after_task4")
ft_i, _ = ms("sequential_ft_t342", 4, "dice"); ft_i2, _ = ms("sequential_ft_t342", 4, "dice", "test_after_task2")
stats = json.load(open(os.path.join(ROOT, "results", "stats_v2.json")))
ri = {r["label"].split(":")[0]: r["mean"] for r in stats["seg"] if "RI" in r["label"]}
take = [(AQUA, "Forgetting is symmetric:", f"under full FT a second segmentation task erases the first in either order (BraTS {ft_b:.2f}→{ft_b2:.2f}, ISLES {ft_i:.2f}→{ft_i2:.2f}); EWC, LwF, replay do not prevent it; frozen strategies cannot forget."),
        (YELLOW, "Pretraining is task-dependent:", f"vs. random init, ΔDice = {ri['BraTS']:+.3f} BraTS, {ri['ISLES']:+.3f} ISLES, {ri['WMH']:+.3f} WMH; run a random-init control."),
        (INK2, "Protocol:", "subject-grouped splits, whole-volume inference, 5 seeds, 301 runs, paired per-case tests with Holm correction.")]
y = 0.075
fig.canvas.draw()
for col, head, body in take:
    fig.patches.append(mp.Rectangle((0.015, y - 0.012), 0.006, 0.024, fc=col, ec="none", transform=fig.transFigure))
    t = fig.text(0.027, y, head, fontsize=8.2, weight="bold", color=INK, va="center")
    bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.text(bb.x1 + 0.004, y, body, fontsize=8.2, color=INK2, va="center")
    y -= 0.033

fig.text(0.015, 0.975, "Which adaptation strategy for a new task?  A rule that is knowable before training: does the task inherit a large head?",
         fontsize=13, weight="bold", color=INK, va="center")

for name, dpi in (("teaser_v4.pdf", 300), ("teaser_v4.png", 300), ("teaser_v4_preview.png", 110)):
    fig.savefig(os.path.join(HERE, name), dpi=dpi, bbox_inches="tight", facecolor="white")
print("written teaser_v4.{pdf,png}")
