#!/usr/bin/env python3
"""Predicted vs true brain age on the held-out test sets, from the per-case predictions
(no inference needed): LoRA + head r=16 against the linear probe, IXI and OASIS-1,
seeds 42-46 pooled.  python scripts/fig_age_scatter.py"""
import glob, json, os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "benchmark_v2", "n32")
BLUE, ORANGE, INK2, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})
def cases(cfg, task):
    pts = []
    for f in sorted(glob.glob(f"{OUT}/{cfg}/seed*/task{task}_per_case.json")):
        pts += [(c["gt"], c["pred"]) for c in json.load(open(f))]
    return np.array(pts)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
for ax, (title, task, suf) in zip(axes, [("IXI brain age (T2, whole head)", 5, "_t5"), ("OASIS-1 brain age (T7)", 9, "_t9")]):
    L, S = cases("lora_head_r16" + suf, task), cases("sequential_linear" + suf, task)
    lo, hi = min(L[:, 0].min(), 15), max(L[:, 0].max(), 95)
    ax.plot([lo, hi], [lo, hi], color=GRID, lw=1, zorder=1)
    ax.scatter(S[:, 0], S[:, 1], s=6, color=ORANGE, alpha=0.35, lw=0, label=f"Linear probe (MAE {np.abs(S[:,1]-S[:,0]).mean():.1f} y, constant)", zorder=2)
    ax.scatter(L[:, 0], L[:, 1], s=6, color=BLUE, alpha=0.45, lw=0, label=f"LoRA + head (MAE {np.abs(L[:,1]-L[:,0]).mean():.1f} y)", zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_xlabel("true age (years)"); ax.set_ylabel("predicted age (years)"); ax.set_title(title, fontsize=9, loc="left", color="#0b0b0b")
    ax.legend(frameon=False, fontsize=7, loc="upper left", markerscale=2.5, handletextpad=0.3)
    ax.text(0.98, 0.03, f"n = {len(L)} test cases, 5 seeds", transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "IEEE_JBHI_3dBrainLoRA", "figures", "fig_age_scatter.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(ROOT, "IEEE_JBHI_3dBrainLoRA", "figures", "fig_age_scatter.png"), dpi=150, bbox_inches="tight"); print("ok")
