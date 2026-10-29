#!/usr/bin/env python3
"""Qualitative segmentation examples on held-out test cases (seed-42 split): ground truth,
LoRA + head (r=16) and the linear probe, on BraTS, ISLES and WMH, with per-case Dice.
Loads the corrected-protocol checkpoints and runs the same sliding-window inference as
the evaluation.  gq run -g 5G -c 2 -m 8 -- python scripts/fig_qualitative_seg.py"""
import importlib.util, json, os, sys
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "fomo_baseline", "src"))
from src.data import TASK_CONFIGS
from src.eval import sliding_window_logits, dice_score
_spec = importlib.util.spec_from_file_location("bench", os.path.join(ROOT, "scripts/run_benchmark_v2.py")); _bench = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_bench)
OUT = os.path.join(ROOT, "outputs", "benchmark_v2", "n32"); PRE = os.path.join(ROOT, "weights/fomo25_mmunetvae_pretrained.ckpt")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RED, BLUE, ORANGE = "#e34948", "#2a78d6", "#eb6834"
# (title, task id, channel to show, LoRA run dir, linear-probe run dir, case index in the seed-42 test split)
PANELS = [("BraTS glioma (T1)", 2, 1, "lora_head_r16", "sequential_linear", 0),
          ("ISLES stroke (T3)", 4, 0, "lora_head_r16_t4", "sequential_linear_t243", 0),
          ("WMH (T6)", 8, 0, "lora_head_r16_t8", "sequential_linear_t8", 0)]
def predict(cfg_dir, task, vol):
    r = json.load(open(f"{OUT}/{cfg_dir}/seed42/results.json")); cfg = TASK_CONFIGS[task]
    lora_r = 16 if cfg_dir.startswith("lora_head_r16") else 8
    model = _bench.build_model(r["method"], task, cfg, (64, 64, 64), PRE, lora_r)
    model.load_state_dict(torch.load(f"{OUT}/{cfg_dir}/seed42/task{task}/task{task}_best.pt", map_location="cpu")["model_state"]); model = model.to(dev).eval()
    with torch.no_grad():
        prob = sliding_window_logits(model, vol, patch=(64, 64, 64), overlap=0.25, batch_size=8, device=dev, max_channels=cfg["modalities"])
    del model; torch.cuda.empty_cache()
    return (prob.argmax(0) == 1).astype(np.uint8)
def square(a):
    h, w = a.shape[:2]; n = max(h, w); o = np.zeros((n, n) + a.shape[2:], a.dtype); o[(n-h)//2:(n-h)//2+h, (n-w)//2:(n-w)//2+w] = a; return o
fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.4))
for row, (title, task, ch, lora_dir, sl_dir, idx) in enumerate(PANELS):
    # show the test case whose LoRA Dice is closest to the median, not a cherry-picked one
    pc = json.load(open(f"{OUT}/{lora_dir}/seed42/task{task}_per_case.json"))
    pc = [c for c in pc if np.isfinite(c["dice"])]; med = float(np.median([c["dice"] for c in pc]))
    base = min(pc, key=lambda c: abs(c["dice"] - med))["case"]
    base = base if os.path.isabs(base) else os.path.join(ROOT, base)
    print(title, "median LoRA Dice", round(med, 3), "case", os.path.basename(base), flush=True)
    vol = np.load(base + ".npy").astype(np.float32); gt = np.load(base + "_seg.npy")[0].astype(np.uint8)
    preds = {"LoRA + head": predict(lora_dir, task, vol), "Linear probe": predict(sl_dir, task, vol)}
    ax_ = int(np.argmin(vol.shape[1:])); k = int(gt.sum(axis=tuple(a for a in range(3) if a != ax_)).argmax())
    img = np.take(vol[ch], k, axis=ax_); fg = img != 0; lo, hi = np.percentile(img[fg], [1, 99]); img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1); img[~fg] = 0
    img = square(np.rot90(img)); gts = square(np.rot90(np.take(gt, k, axis=ax_)))
    for col, (name, mask3d, color) in enumerate([("Ground truth", gt, RED), ("LoRA + head", preds["LoRA + head"], BLUE), ("Linear probe", preds["Linear probe"], ORANGE)]):
        ax = axes[row, col]; ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear")
        m = square(np.rot90(np.take(mask3d, k, axis=ax_)))
        ov = np.zeros(img.shape + (4,)); ov[m > 0] = matplotlib.colors.to_rgba(color, 0.55); ax.imshow(ov, interpolation="nearest")
        if col > 0:  # GT outline for reference + per-case Dice over the whole volume
            ax.contour(gts, levels=[0.5], colors=[RED], linewidths=0.6)
            ax.text(0.03, 0.04, f"Dice {dice_score(mask3d, gt):.3f}", transform=ax.transAxes, color="white", fontsize=8, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#0b0b0b", ec="none", alpha=0.7))
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0: ax.set_title(name, fontsize=9)
        if col == 0: ax.set_ylabel(title, fontsize=9)
    print(title, {k: round(float(dice_score(v, gt)), 3) for k, v in preds.items()}, flush=True)
fig.tight_layout(h_pad=0.6, w_pad=0.4)
fig.savefig(os.path.join(ROOT, "IEEE_JBHI_3dBrainLoRA/figures/fig_qualitative_seg.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(ROOT, "IEEE_JBHI_3dBrainLoRA/figures/fig_qualitative_seg.png"), dpi=150, bbox_inches="tight"); print("ok")
