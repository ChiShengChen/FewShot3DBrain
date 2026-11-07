"""
Fig. 1 (teaser) as a fully vector PDF, drawn after the accepted image-model draft
(figures/teaser_gen_v3.png) so it can be printed at any resolution:
  (a) the nine tasks as real held-out slices, grouped by task-head size
  (b) the four adaptation strategies as what is frozen (snowflake) and trained (flame)
  (c) the decision rule, read from outputs/benchmark_v2 and results/stats_v2.json
Only the MRI slices are raster (embedded at native resolution); everything else is paths.

  python make_teaser_v5.py          -> figures/teaser_v5.{pdf,png}, teaser_v5_preview.png
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RES = os.path.join(ROOT, "outputs", "benchmark_v2", "n32")

# palette (paper series colours + the draft's navy / light-blue chrome)
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
NAVY, INK, INK2, MUTED, LINE = "#1f3f8f", "#111111", "#333333", "#777777", "#d9d9d9"
PANEL_EC, ROW_FC, ROW_EC = "#9fc0e6", "#f6f9fe", "#bcd4ee"
FROZ = dict(front="#d6e7f9", top="#ebf4fd", side="#b3cfee", edge="#4f8ccc")
TRAIN = dict(front="#fde3cf", top="#fff1e6", side="#f6c39b", edge="#e8763a")
import matplotlib.font_manager as fm
for _f in glob.glob("/usr/share/fonts/truetype/lato/Lato-*.ttf"):   # conda's matplotlib does not see system fonts
    fm.fontManager.addfont(_f)
plt.rcParams.update({"font.family": ["Lato", "DejaVu Sans"], "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})

W, H = 16.0, 6.9           # inches; everything below is placed in inch coordinates
TOP, BOT = H - 0.62, 0.98  # panel band
FOOT0, FOOT1 = 0.12, 0.78  # footer band


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


# ── drawing primitives (all in inches on one aspect-equal axes) ───────────────
fig = plt.figure(figsize=(W, H), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")


def rbox(x, y, w, h, fc="none", ec=PANEL_EC, lw=1.2, r=0.12, ls="-", z=1):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def txt(x, y, s, fs=9, color=INK, weight="normal", ha="left", va="center", z=5, **kw):
    return ax.text(x, y, s, fontsize=fs, color=color, weight=weight, ha=ha, va=va, zorder=z, **kw)


def arrow(x0, y0, x1, y1, lw=1.0, color=INK2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=4,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=0, shrinkB=0, mutation_scale=10))


def cube(x, y, s, pal, d=0.13, z=3):
    """Isometric block: front face at (x, y) with side s, depth d."""
    ax.add_patch(mp.Polygon([(x, y + s), (x + d, y + s + d), (x + s + d, y + s + d), (x + s, y + s)], fc=pal["top"], ec=pal["edge"], lw=0.8, zorder=z))
    ax.add_patch(mp.Polygon([(x + s, y), (x + s + d, y + d), (x + s + d, y + s + d), (x + s, y + s)], fc=pal["side"], ec=pal["edge"], lw=0.8, zorder=z))
    ax.add_patch(mp.Rectangle((x, y), s, s, fc=pal["front"], ec=pal["edge"], lw=0.8, zorder=z + 1))


def flame(x, y, s=0.1, z=8):
    pts = np.array([[0, -1], [0.55, -0.6], [0.45, 0.1], [0.75, 0.35], [0.25, 1.0], [-0.1, 0.45], [-0.45, 0.9], [-0.7, 0.2], [-0.5, -0.6]]) * s
    ax.add_patch(mp.Polygon(pts + [x, y], closed=True, fc="#f0652b", ec="#c74a12", lw=0.5, zorder=z))
    ax.add_patch(mp.Polygon(pts * 0.5 + [x, y - 0.3 * s], closed=True, fc="#ffc83d", ec="none", zorder=z + 1))


def snow(x, y, s=0.09, color="#3b7fd0", z=8):
    for k in range(6):
        a = np.deg2rad(60 * k); dx, dy = np.cos(a), np.sin(a)
        ax.plot([x, x + s * dx], [y, y + s * dy], color=color, lw=1.1, solid_capstyle="round", zorder=z)
        for sgn in (1, -1):   # one pair of branches per spoke
            b = a + sgn * np.deg2rad(60); bx, by = x + 0.55 * s * dx, y + 0.55 * s * dy
            ax.plot([bx, bx + 0.35 * s * np.cos(b)], [by, by + 0.35 * s * np.sin(b)], color=color, lw=0.9, solid_capstyle="round", zorder=z)


def lora_branch(bx, by, s):
    """Trapezoid adapter 'A' under a block, summed back into the stream with ⊕."""
    tx, ty, tw, th = bx + 0.0, by - 0.3, 0.24, 0.22
    ax.add_patch(mp.Polygon([(tx, ty), (tx + tw, ty + 0.05), (tx + tw, ty + th - 0.05), (tx, ty + th)], fc=TRAIN["front"], ec=TRAIN["edge"], lw=0.8, zorder=4))
    txt(tx + tw / 2 - 0.01, ty + th / 2, "A", fs=7, color=INK, ha="center")
    cx, cy = bx + s - 0.03, ty + th / 2
    ax.add_patch(mp.Circle((cx, cy), 0.065, fc="white", ec=INK2, lw=0.9, zorder=5))
    ax.plot([cx - 0.04, cx + 0.04], [cy, cy], color=INK2, lw=0.9, zorder=6); ax.plot([cx, cx], [cy - 0.04, cy + 0.04], color=INK2, lw=0.9, zorder=6)
    ax.plot([tx + tw, cx - 0.065], [cy, cy], color=INK2, lw=0.8, zorder=4)
    ax.plot([cx, cx], [cy + 0.065, by], color=INK2, lw=0.8, zorder=4)


def bias_slot(bx, by, s):
    ax.add_patch(mp.Rectangle((bx + s - 0.15, by + 0.04), 0.11, 0.11, fc=TRAIN["front"], ec=TRAIN["edge"], lw=0.8, zorder=6))


def mri_cube(cx, cy, s, img):
    """Dark cube with a real slice on the front face (the network input)."""
    d = 0.1; x, y = cx - s / 2, cy - s / 2 - 0.03
    ax.add_patch(mp.Polygon([(x, y + s), (x + d, y + s + d), (x + s + d, y + s + d), (x + s, y + s)], fc="#3c3c3c", ec="#222222", lw=0.6, zorder=3))
    ax.add_patch(mp.Polygon([(x + s, y), (x + s + d, y + d), (x + s + d, y + s + d), (x + s, y + s)], fc="#2a2a2a", ec="#222222", lw=0.6, zorder=3))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=(x, x + s, y, y + s), interpolation="bilinear", zorder=4)
    ax.add_patch(mp.Rectangle((x, y), s, s, fc="none", ec="#222222", lw=0.6, zorder=5))


# ── title ─────────────────────────────────────────────────────────────────────
txt(W / 2, H - 0.3, "Which adaptation strategy for a new task?  What the task inherits decides it: train the large head; fit the small one properly before moving the encoder",
    fs=15, color=NAVY, weight="bold", ha="center")

# ── panel frames ──────────────────────────────────────────────────────────────
PA = (0.12, 5.05); PB = (5.2, 10.88); PC = (11.03, 15.88)
for x0, x1 in (PA, PB, PC):
    rbox(x0, BOT, x1 - x0, TOP - BOT, ec=PANEL_EC, lw=1.4, r=0.14)


def panel_head(x0, letter, title):
    txt(x0 + 0.18, TOP - 0.3, f"({letter})", fs=14, color=NAVY, weight="bold")
    txt(x0 + 0.68, TOP - 0.3, title, fs=12, color=NAVY, weight="bold")


# (a) tasks ---------------------------------------------------------------------
panel_head(PA[0], "a", "Nine tasks, six cohorts, 32 labelled volumes each")
seg = [("BraTS glioma", "Task002_FOMO2", 1), ("ISLES stroke", "Task004_ISLES_SEG", 0), ("Hippocampus", "Task007_HIPPO", 0), ("WMH, 5 scanners", "Task008_WMH_SEG", 0)]
glob_ = [("IXI age", "Task005_IXI_WB", 0), ("IXI sex", "Task006_IXI_SEX", 0), ("OASIS age", "Task009_OASIS_AGE", 0), ("OASIS sex", "Task010_OASIS_SEX", 0), ("OASIS CDR", "Task011_OASIS_CDR", 0)]


def tiles(items, ytop, with_mask, color, head):
    gx0, gx1 = PA[0] + 0.14, PA[1] - 0.14
    gap, pad = 0.1, 0.16
    w = (gx1 - gx0 - 2 * pad - (len(items) - 1) * gap) / len(items)
    h = 0.42 + w + 0.3
    rbox(gx0, ytop - h, gx1 - gx0, h, ec=color, lw=1.3, r=0.12, ls=(0, (5, 3)))
    txt(gx0 + pad, ytop - 0.22, head, fs=10.5, color=color, weight="bold")
    for i, (name, tname, ch) in enumerate(items):
        x = gx0 + pad + i * (w + gap); y = ytop - 0.42 - w
        img, m = slice_of(test_case(tname), ch, with_mask)
        ext = (x, x + w, y, y + w)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="bilinear", zorder=2)
        if m is not None:
            ov = np.zeros(img.shape + (4,)); ov[m > 0] = matplotlib.colors.to_rgba(RED, 0.6)
            ax.imshow(ov, extent=ext, interpolation="nearest", zorder=3)
        ax.add_patch(mp.Rectangle((x, y), w, w, fc="none", ec="#222222", lw=0.6, zorder=4))
        txt(x + w / 2, y - 0.15, name, fs=8.6, color=INK2, ha="center")
    return ytop - h


y1 = tiles(seg, TOP - 0.85, True, ORANGE, "Large task head: inherits the 8.45M-parameter decoder")
y2 = tiles(glob_, y1 - 0.42, False, BLUE, "Small task head: ~1K-parameter head on pooled features")
txt(PA[0] + 0.3, (y2 + BOT) / 2, "One held-out test slice per task; segmentation targets in red.  Six cohorts, five scanners;\nthe head size is a property of the task that is known before any training run.", fs=8, color=MUTED, va="center", linespacing=1.35)

# (b) strategies ----------------------------------------------------------------
panel_head(PB[0], "b", "Adaptation strategies: what is frozen and what is trained")
S, D = 0.42, 0.11                                   # block side, depth
BX = [6.58, 7.33, 8.25, 9.0]                        # block x positions: encoder ×2, decoder ×2
HX, HW = 9.82, 0.46                                 # head box
RX0, RX1 = PB[0] + 0.1, PB[1] - 0.1
HY = TOP - 0.82
txt(RX0 + 0.5, HY, "input", fs=10.5, color=NAVY, weight="bold", ha="center")
txt((BX[0] + BX[1] + S + D) / 2, HY, "encoder", fs=10.5, color=NAVY, weight="bold", ha="center")
txt((BX[2] + BX[3] + S + D) / 2, HY, "decoder", fs=10.5, color=NAVY, weight="bold", ha="center")
txt(HX + HW / 2, HY, "head", fs=10.5, color=NAVY, weight="bold", ha="center")
rows = [("LoRA + head", "lora", 1.06), ("Linear probe", "probe", 0.78), ("BitFit", "bitfit", 0.78), ("Full fine-tuning", "ft", 0.78)]
in_img, _ = slice_of(test_case("Task005_IXI_WB"), 0, False)
y = HY - 0.2
for label, kind, rh in rows:
    rbox(RX0, y - rh, RX1 - RX0, rh, fc=ROW_FC, ec=ROW_EC, lw=1.0, r=0.1)
    txt(RX0 + 0.12, y - 0.17, label, fs=11, color=NAVY, weight="bold")
    yb = y - 0.28 - S                                         # block bottom: under the name row
    mri_cube(RX0 + 0.5, yb + S / 2, 0.38, in_img)
    arrow(RX0 + 0.8, yb + S / 2, BX[0] - 0.04, yb + S / 2)
    enc_train, dec_train = kind == "ft", kind in ("probe", "ft")
    for i, bx in enumerate(BX):
        trained = enc_train if i < 2 else dec_train
        cube(bx, yb, S, TRAIN if trained else FROZ, D)
        (flame if trained else snow)(bx + S / 2, yb + S / 2 + 0.02, 0.085)
        if kind == "lora": lora_branch(bx, yb, S)
        if kind == "bitfit": bias_slot(bx, yb, S)
        if i < 3: arrow(bx + S + D + 0.02, yb + S / 2, BX[i + 1] - 0.04, yb + S / 2)
    rbox(HX, yb + 0.02, HW, S - 0.04, fc=TRAIN["front"], ec=TRAIN["edge"], lw=0.9, r=0.06, z=3)
    txt(HX + HW / 2, yb + S / 2 - 0.03, "Head", fs=8.5, color=INK, ha="center")
    flame(HX + HW - 0.08, yb + S - 0.08, 0.06)
    arrow(BX[3] + S + D + 0.02, yb + S / 2, HX - 0.04, yb + S / 2)
    arrow(HX + HW + 0.02, yb + S / 2, RX1 - 0.3, yb + S / 2)
    txt(RX1 - 0.16, yb + S / 2, "ŷ", fs=11, color=INK, ha="center", weight="bold")
    y -= rh + 0.09
# legend
ly = BOT + 0.2
snow(RX0 + 0.2, ly, 0.08); txt(RX0 + 0.34, ly, "= frozen", fs=8, color=INK2)
flame(RX0 + 1.12, ly, 0.07); txt(RX0 + 1.24, ly, "= trained for the new task", fs=8, color=INK2)
ax.add_patch(mp.Polygon([(RX0 + 2.92, ly - 0.09), (RX0 + 3.12, ly - 0.06), (RX0 + 3.12, ly + 0.06), (RX0 + 2.92, ly + 0.09)], fc=TRAIN["front"], ec=TRAIN["edge"], lw=0.7, zorder=4))
txt(RX0 + 3.01, ly, "A", fs=6.5, ha="center"); txt(RX0 + 3.2, ly, "= low-rank adapter (LoRA)", fs=8, color=INK2)
ax.add_patch(mp.Rectangle((RX0 + 4.6, ly - 0.06), 0.12, 0.12, fc=TRAIN["front"], ec=TRAIN["edge"], lw=0.7, zorder=4))
txt(RX0 + 4.8, ly, "= trainable bias (BitFit)", fs=8, color=INK2)
txt(RX0 + 0.06, ly + 0.26, "Input: 3D MRI, C × 64³ patch (96³ whole head for age / sex).   Backbone: FOMO 3D U-Net; 32 volumes, 100 epochs, one recipe.", fs=7, color=MUTED, ha="left")

# (c) results -------------------------------------------------------------------
txt(PC[0] + 0.18, TOP - 0.3, "(c)", fs=14, color=NAVY, weight="bold")
methods = [("LoRA + head", BLUE), ("Linear probe", ORANGE), ("BitFit", YELLOW), ("Full FT", AQUA)]
cfg = {"LoRA + head": "lora_head_r16", "Linear probe": "sequential_linear", "BitFit": "bitfit", "Full FT": "sequential_ft"}


def chart(y0, y1):
    a = fig.add_axes([(PC[0] + 0.62) / W, y0 / H, (PC[1] - PC[0] - 0.85) / W, (y1 - y0) / H])
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.spines["left"].set_color(LINE); a.spines["bottom"].set_color(LINE); a.tick_params(colors=INK2, length=0, labelsize=8)
    a.grid(axis="y", color="#ececec", lw=0.7); a.set_axisbelow(True)
    return a


bw = 0.19
a1 = chart(3.85, 5.55)
tasks = [("BraTS", "", 2), ("ISLES", "_t4", 4), ("Hippo.", "_t7", 7), ("WMH", "_t8", 8)]
for mi, (mname, col) in enumerate(methods):
    xs, ys, es = [], [], []
    for ti, (tl, suf, task) in enumerate(tasks):
        c = cfg[mname] + suf
        m, s = ms(c, task, "dice"); xs.append(ti + (mi - 1.5) * bw); ys.append(m); es.append(s)
    a1.bar(xs, ys, bw - 0.03, color=col, label=mname, zorder=2); a1.errorbar(xs, ys, es, fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1.5, zorder=3)
a1.set_xticks(range(4)); a1.set_xticklabels([t[0] for t in tasks]); a1.set_ylim(0, 1.0); a1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); a1.set_ylabel("Dice", fontsize=9, color=INK2)
a1.set_title("Large head → train it; LoRA matches it, never beats it; BitFit collapses", fontsize=9.5, weight="bold", loc="left", color=INK, pad=7)

a2 = chart(1.62, 3.3)
gtasks = [("IXI age", 5, "_t5", "r2"), ("OASIS age", 9, "_t9", "r2"), ("IXI sex", 6, "_t6", "auc"), ("OASIS sex", 10, "_t10", "auc")]
methods2 = methods + [("Closed-form probe", RED)]
cfg2 = dict(cfg, **{"Closed-form probe": "ridge_probe"})
bw2 = 0.155
for mi, (mname, col) in enumerate(methods2):
    xs, ys, es = [], [], []
    for ti, (tl, task, suf, key) in enumerate(gtasks):
        m, s = ms(cfg2[mname] + suf, task, key); xs.append(ti + (mi - 2) * bw2); ys.append(max(m, 0) if key == "r2" else m); es.append(s)
    a2.bar(xs, ys, bw2 - 0.025, color=col, label=mname if mname == "Closed-form probe" else None, zorder=2); a2.errorbar(xs, ys, es, fmt="none", ecolor=INK2, elinewidth=0.6, capsize=1.5, zorder=3)
a2.axhline(0.5, xmin=0.44, xmax=1.0, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1); a2.text(1.5, 0.5, "chance", fontsize=6.8, color=MUTED, ha="center", va="center", zorder=4, bbox=dict(fc="white", ec="none", pad=1.2))
a2.set_xticks(range(4)); a2.set_xticklabels([t[0] for t in gtasks]); a2.set_ylim(0, 1.0); a2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); a2.set_ylabel("R²  |  AUC", fontsize=9, color=INK2)
a2.set_title("Small head → a closed-form probe learns from frozen features;\nthe gradient-trained probe and BitFit output a constant", fontsize=9.5, weight="bold", loc="left", color=INK, pad=7)
h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a2.legend(h1 + h2, l1 + l2, ncol=5, handletextpad=0.4, fontsize=7.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.19), handlelength=1.0, columnspacing=0.9)

# ── footer: three findings ─────────────────────────────────────────────────────
ft_b, _ = ms("sequential_ft_t243", 2, "dice"); ft_b2, _ = ms("sequential_ft_t243", 2, "dice", "test_after_task4")
ft_i, _ = ms("sequential_ft_t342", 4, "dice"); ft_i2, _ = ms("sequential_ft_t342", 4, "dice", "test_after_task2")
stats = json.load(open(os.path.join(ROOT, "results", "stats_v2.json")))
ri = {r["label"].split(":")[0]: r["mean"] for r in stats["seg"] if "RI" in r["label"]}
nruns = len([f for n in ("n16", "n32", "n64", "nfull") for f in glob.glob(os.path.join(ROOT, "outputs", "benchmark_v2", n, "*", "seed*", "results.json")) if "smoke" not in f])
sgn = lambda v: f"{v:+.3f}".replace("-", "−")
take = [(ORANGE, "Forgetting:", f"a second segmentation task erases the first in either\norder (BraTS {ft_b:.2f}→{ft_b2:.2f}); EWC softens it, only freezing prevents it."),
        (BLUE, "Pretraining:", f"59 of 82 tensors load; under LoRA it equals random\ninit on four tasks ({sgn(ri['BraTS'])} Dice on BraTS); frozen features beat random."),
        (AQUA, "Protocol:", f"subject-grouped splits, whole-volume inference,\n5 seeds, {nruns} runs, seed-level paired tests, per-strategy learning rate.")]
rbox(PA[0], FOOT0, PC[1] - PA[0], FOOT1 - FOOT0, ec=PANEL_EC, lw=1.4, r=0.12)
cw = (PC[1] - PA[0]) / 3
fig.canvas.draw()
for i, (col, head, body) in enumerate(take):
    x = PA[0] + i * cw
    if i: ax.plot([x, x], [FOOT0 + 0.1, FOOT1 - 0.1], color=PANEL_EC, lw=1.0, zorder=2)
    ax.add_patch(mp.FancyBboxPatch((x + 0.22, (FOOT0 + FOOT1) / 2 - 0.1), 0.2, 0.2, boxstyle="round,pad=0,rounding_size=0.03", fc=col, ec="none", zorder=3))
    t = txt(x + 0.55, FOOT1 - 0.24, head, fs=9.2, color=NAVY, weight="bold")
    bb = t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    l1, l2 = body.split("\n")
    txt(bb.x1 + 0.05, FOOT1 - 0.24, l1, fs=9.2, color=INK2)
    txt(x + 0.55, FOOT1 - 0.46, l2, fs=9.2, color=INK2)

out = os.path.join(HERE, "figures")
for name, dpi in (("teaser_v5.pdf", 300), ("teaser_v5.png", 300), ("teaser_v5_preview.png", 120)):
    fig.savefig(os.path.join(out, name), dpi=dpi, facecolor="white")
print(f"written figures/teaser_v5.{{pdf,png}}  ({nruns} runs)")
