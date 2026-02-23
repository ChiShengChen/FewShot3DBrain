#!/usr/bin/env python3
"""
Visualize LoRA: Task 2 (segmentation) and Task 3 (brain age).
Usage:
  python scripts/visualize_segmentation.py --task all --seeds 42 43 44 --n_samples 4
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))

from src.data import TASK_CONFIGS, get_few_shot_dataloaders
from src.models import TaskModel

import torch


def load_model_task2(checkpoint_path: str, pretrained_path: str, device):
    """Load TaskModel for T2 (segmentation)."""
    cfg = TASK_CONFIGS[2]
    model = TaskModel(
        task_id=2, input_channels=cfg["modalities"], patch_size=(64, 64, 64),
        pretrained_path=pretrained_path, model_name="unet_b", lora_r=8, lora_decoder=True,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval()


def load_model_task3(checkpoint_path: str, pretrained_path: str, device):
    """Load TaskModel for T3 (brain age regression)."""
    cfg = TASK_CONFIGS[3]
    model = TaskModel(
        task_id=3, input_channels=cfg["modalities"], patch_size=(64, 64, 64),
        pretrained_path=pretrained_path, model_name="unet_b", lora_r=8, lora_decoder=False,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval()


def viz_slice(img_slice, gt_slice, pred_slice, out_path, suptitle="LoRA tumor segmentation (BraTS)"):
    """Plot Input | GT | Pred | Overlay in one row."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes[0].imshow(img_slice, cmap="gray")
    axes[0].set_title("Input (T2-FLAIR)")
    axes[0].axis("off")
    axes[1].imshow(img_slice, cmap="gray")
    axes[1].imshow(gt_slice, cmap="Reds", alpha=0.5)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    axes[2].imshow(img_slice, cmap="gray")
    axes[2].imshow(pred_slice, cmap="Greens", alpha=0.5)
    axes[2].set_title("LoRA Prediction")
    axes[2].axis("off")
    axes[3].imshow(img_slice, cmap="gray")
    axes[3].imshow(gt_slice, cmap="Reds", alpha=0.4)
    axes[3].imshow(pred_slice, cmap="Greens", alpha=0.4)
    axes[3].set_title("Overlay (GT red, Pred green)")
    axes[3].axis("off")
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def viz_slice_stack6(rows, out_path, suptitle=None):
    """Plot N different samples with GT: each row = Input | GT | Pred | Overlay."""
    n = len(rows)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(12, 2.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for i, (img_slice, gt_slice, pred_slice) in enumerate(rows):
        axes[i, 0].imshow(img_slice, cmap="gray")
        axes[i, 0].set_ylabel(f"Sample {i+1}", fontsize=9)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(img_slice, cmap="gray")
        axes[i, 1].imshow(gt_slice, cmap="Reds", alpha=0.5)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(img_slice, cmap="gray")
        axes[i, 2].imshow(pred_slice, cmap="Greens", alpha=0.5)
        axes[i, 2].axis("off")
        axes[i, 3].imshow(img_slice, cmap="gray")
        axes[i, 3].imshow(gt_slice, cmap="Reds", alpha=0.4)
        axes[i, 3].imshow(pred_slice, cmap="Greens", alpha=0.4)
        axes[i, 3].axis("off")
    for j, col_title in enumerate(["Input (T2-FLAIR)", "Ground Truth", "LoRA Prediction", "Overlay"]):
        axes[0, j].set_title(col_title, fontsize=10)
    title = (suptitle or "LoRA tumor segmentation (BraTS)") + f" ({n} samples with tumor GT)"
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _norm_for_display(arr, p_low=1, p_high=99):
    """Normalize for display. Handles z-scored data (can be negative)."""
    arr = np.asarray(arr, dtype=np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(valid, [p_low, p_high])
    if lo >= hi:
        lo, hi = valid.min(), valid.max()
    if lo >= hi:
        return np.zeros_like(arr) + 0.5  # constant -> mid gray
    arr = np.clip(arr, lo, hi)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)


def viz_task3(img, pred_age, gt_age, out_path):
    """Task 3 (Regression): MRI views + regression results panel."""
    from scipy.ndimage import zoom
    d, h, w = img.shape
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    # 3 orthogonal slices (resize to same size for uniform display)
    sz = 64
    s1 = _norm_for_display(img[d//2])
    s2 = _norm_for_display(np.transpose(img[:, h//2, :], (1, 0)))
    s3 = _norm_for_display(np.transpose(img[:, :, w//2], (1, 0)))
    for ax, slc, title in zip(axes[:3], [s1, s2, s3], ["Axial", "Coronal", "Sagittal"]):
        if slc.shape[0] != sz or slc.shape[1] != sz:
            z = (sz / slc.shape[0], sz / slc.shape[1])
            slc = zoom(slc, z, order=1)
        ax.imshow(slc, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    # Regression results panel
    axes[3].axis("off")
    err = abs(pred_age - gt_age)
    axes[3].text(0.5, 0.9, "Brain Age\n(Regression)", ha="center", fontsize=11, fontweight="bold")
    axes[3].text(0.5, 0.65, f"Predicted:  {pred_age:.1f} yr", ha="center", fontsize=10)
    axes[3].text(0.5, 0.50, f"Actual:     {gt_age:.1f} yr", ha="center", fontsize=10)
    axes[3].text(0.5, 0.35, f"|Error|:    {err:.1f} yr", ha="center", fontsize=10,
                 color="green" if err < 2 else "orange")
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)
    plt.suptitle("Task 3: LoRA brain age regression (IXI)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def viz_task3_stack6(rows, out_path, suptitle=None):
    """Task 3 stacked: 6 rows, each = Axial | Coronal | Sagittal | Regression panel."""
    from scipy.ndimage import zoom
    n = len(rows)
    if n == 0:
        return
    sz = 64
    fig, axes = plt.subplots(n, 4, figsize=(12, 2.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for j, col_title in enumerate(["Axial", "Coronal", "Sagittal", "Regression"]):
        axes[0, j].set_title(col_title, fontsize=10)
    for i, (img, pred_age, gt_age) in enumerate(rows):
        d, h, w = img.shape
        s1 = _norm_for_display(img[d//2])
        s2 = _norm_for_display(np.transpose(img[:, h//2, :], (1, 0)))
        s3 = _norm_for_display(np.transpose(img[:, :, w//2], (1, 0)))
        for k, slc in enumerate([s1, s2, s3]):
            ax = axes[i, k]
            if slc.shape[0] != sz or slc.shape[1] != sz:
                z = (sz / slc.shape[0], sz / slc.shape[1])
                slc = zoom(slc, z, order=1)
            ax.imshow(slc, cmap="gray", vmin=0, vmax=1)
            if k == 0:
                ax.set_ylabel(f"Sample {i+1}", fontsize=9)
            ax.axis("off")
        axes[i, 3].axis("off")
        err = abs(pred_age - gt_age)
        axes[i, 3].text(0.5, 0.85, f"Pred: {pred_age:.1f} yr", ha="center", fontsize=9)
        axes[i, 3].text(0.5, 0.55, f"GT:  {gt_age:.1f} yr", ha="center", fontsize=9)
        axes[i, 3].text(0.5, 0.25, f"|Err|: {err:.1f}", ha="center", fontsize=9,
                       color="green" if err < 2 else "orange")
        axes[i, 3].set_xlim(0, 1)
        axes[i, 3].set_ylim(0, 1)
    fig.suptitle(suptitle or "LoRA brain age regression (IXI)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _dice_slice(gt_sl, pred_sl, thresh=0.5):
    """Dice overlap on 2D slice."""
    a = (gt_sl > thresh).astype(np.float32)
    b = (pred_sl > thresh).astype(np.float32)
    inter = (a * b).sum()
    return 2 * inter / (a.sum() + b.sum() + 1e-8)


def run_task2(args, device):
    """Generate Task 2 (segmentation) visualizations."""
    base = args.checkpoint_base or "./outputs/multi_seed/lora"
    seeds = args.seeds or [42, 43, 44]
    total = 0
    stack_rows = []
    for seed in seeds:
        ckpt = f"{base}/seed{seed}/adapters/task2/task2_best.pt"
        if not os.path.exists(ckpt):
            print(f"Skip seed {seed}: {ckpt} not found")
            continue
        print(f"Task 2 seed={seed}: loading {ckpt}")
        model = load_model_task2(ckpt, args.pretrained_path, device)
        _, val_loader = get_few_shot_dataloaders(
            args.data_dir, task_id=2, n_shot=args.n_shot,
            patch_size=(64, 64, 64), batch_size=1, seed=seed,
        )
        n_saved = 0
        candidates = []  # (dice, img_sl, gt_sl, pred_sl) for --stack
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"]
                out = model(x)
                pred = (out.softmax(1)[:, 1] > 0.5).float().cpu().numpy()[0]
                img = x[0, 1].cpu().numpy()
                gt = y[0, 0].float().cpu().numpy()
                d = img.shape[0]
                gt_sl = gt[d//2]
                pred_sl = pred[d//2]
                has_gt_on_slice = (gt_sl > 0.5).any()
                if n_saved < args.n_samples:
                    out_path = os.path.join(args.out_dir, f"task2_seg_s{seed}_{n_saved}.png")
                    viz_slice(img[d//2], gt_sl, pred_sl, out_path)
                    print(f"  Saved {out_path}")
                    total += 1
                if args.stack and has_gt_on_slice:
                    dice = _dice_slice(gt_sl, pred_sl)
                    candidates.append((dice, img[d//2].copy(), gt_sl.copy(), pred_sl.copy()))
                n_saved += 1
                if n_saved >= args.n_samples and not args.stack:
                    break
        if args.stack and len(candidates) >= args.stack:
            reverse = not args.worst  # best: high Dice; worst: low Dice
            candidates.sort(key=lambda t: t[0], reverse=reverse)
            best = candidates[: args.stack]
            stack_rows = [(r[1], r[2], r[3]) for r in best]
            suffix = "worst" if args.worst else ""
            out_stack = os.path.join(args.out_dir, f"task2_seg_s{seed}_stack{args.stack}{suffix}.png")
            viz_slice_stack6(stack_rows, out_stack, suptitle="LoRA under-segmentation (BraTS)" if args.worst else None)
            dices = [f"{r[0]:.2f}" for r in best]
            print(f"  Saved {out_stack} ({'lowest' if args.worst else 'top'} Dice: {dices})")
    return total


def run_task3(args, device):
    """Generate Task 3 (brain age) visualizations."""
    base = args.checkpoint_base or "./outputs/multi_seed/lora"
    seeds = args.seeds or [42, 43, 44]
    total = 0
    for seed in seeds:
        ckpt = f"{base}/seed{seed}/adapters/task3/task3_best.pt"
        if not os.path.exists(ckpt):
            print(f"Skip seed {seed}: {ckpt} not found")
            continue
        print(f"Task 3 seed={seed}: loading {ckpt}")
        model = load_model_task3(ckpt, args.pretrained_path, device)
        _, val_loader = get_few_shot_dataloaders(
            args.data_dir, task_id=3, n_shot=args.n_shot,
            patch_size=(64, 64, 64), batch_size=1, seed=seed,
        )
        n_saved = 0
        candidates = []  # (err, img, pred_age, gt_age) for --stack
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"]
                out = model(x)
                pred_age = float(out.cpu().numpy()[0, 0])
                gt_age = float(y.cpu().numpy().flatten()[0])
                img = x[0].cpu().numpy()  # (C,D,H,W)
                if img.ndim == 4:
                    img = np.mean(img, axis=0)
                if n_saved < args.n_samples:
                    out_path = os.path.join(args.out_dir, f"task3_age_s{seed}_{n_saved}.png")
                    viz_task3(img.copy(), pred_age, gt_age, out_path)
                    print(f"  Saved {out_path} (pred={pred_age:.1f}, gt={gt_age:.1f})")
                    total += 1
                if args.stack:
                    err = abs(pred_age - gt_age)
                    candidates.append((err, img.copy(), pred_age, gt_age))
                n_saved += 1
                if n_saved >= args.n_samples and not args.stack:
                    break
        if args.stack and len(candidates) >= args.stack:
            reverse = args.worst  # best: low err; worst: high err
            candidates.sort(key=lambda t: t[0], reverse=reverse)
            best = candidates[: args.stack]
            rows = [(r[1], r[2], r[3]) for r in best]
            suffix = "worst" if args.worst else ""
            out_stack = os.path.join(args.out_dir, f"task3_age_s{seed}_stack{args.stack}{suffix}.png")
            viz_task3_stack6(rows, out_stack, suptitle="LoRA age underestimation (IXI)" if args.worst else None)
            errs = [f"{r[0]:.1f}" for r in best]
            print(f"  Saved {out_stack} ({'highest' if args.worst else 'lowest'} |Error|: {errs})")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=["2", "3", "all"])
    parser.add_argument("--checkpoint_base", default="./outputs/multi_seed/lora")
    parser.add_argument("--data_dir", default="./data/preprocessed")
    parser.add_argument("--pretrained_path", default="./weights/fomo25_mmunetvae_pretrained.ckpt")
    parser.add_argument("--out_dir", default="./outputs/figures")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--n_shot", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--stack", type=int, default=0, help="Generate stacked figure with N different samples (e.g. 6 for paper review)")
    parser.add_argument("--worst", action="store_true", help="Select worst samples (low Dice, high MAE) for failure case appendix")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    total = 0
    if args.task in ("2", "all"):
        total += run_task2(args, device)
    if args.task in ("3", "all"):
        total += run_task3(args, device)

    print(f"Done. Saved {total} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
