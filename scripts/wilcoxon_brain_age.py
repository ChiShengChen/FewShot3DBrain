#!/usr/bin/env python3
"""
Run Wilcoxon signed-rank test on brain age residuals (predicted - true age).
Reports p-value for H0: median residual = 0 (no systematic bias).

Usage:
  python scripts/wilcoxon_brain_age.py --seed 42 --n_shot 32
  python scripts/wilcoxon_brain_age.py --seed 42 43 44  # aggregate over seeds
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))

from src.data import TASK_CONFIGS, get_few_shot_dataloaders
from src.models import TaskModel

import torch
from scipy import stats


def load_model_task3(checkpoint_path: str, pretrained_path: str, device):
    """Load TaskModel for brain age regression (task_id=3)."""
    cfg = TASK_CONFIGS[3]
    model = TaskModel(
        task_id=3,
        input_channels=cfg["modalities"],
        patch_size=(64, 64, 64),
        pretrained_path=pretrained_path,
        model_name="unet_b",
        lora_r=8,
        lora_decoder=False,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model.to(device).eval()


def collect_predictions(model, val_loader, device):
    """Collect (pred_age, gt_age) for all validation samples."""
    preds, gts = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["label"]
            out = model(x)
            pred_age = float(out.cpu().numpy()[0, 0])
            gt_age = float(y.cpu().numpy().flatten()[0])
            preds.append(pred_age)
            gts.append(gt_age)
    return np.array(preds), np.array(gts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_base", default="./outputs/multi_seed/lora")
    parser.add_argument("--data_dir", default="./data/preprocessed")
    parser.add_argument("--pretrained_path", default="./weights/fomo25_mmunetvae_pretrained.ckpt")
    parser.add_argument("--seed", type=int, nargs="+", default=[42])
    parser.add_argument("--n_shot", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_residuals = []
    all_preds, all_gts = [], []

    for seed in args.seed:
        ckpt = Path(args.checkpoint_base) / f"seed{seed}" / "adapters" / "task3" / "task3_best.pt"
        if not ckpt.exists():
            print(f"Skip seed {seed}: {ckpt} not found")
            continue
        print(f"Seed {seed}: loading {ckpt}")
        model = load_model_task3(str(ckpt), args.pretrained_path, device)
        _, val_loader = get_few_shot_dataloaders(
            args.data_dir,
            task_id=3,
            n_shot=args.n_shot,
            patch_size=(64, 64, 64),
            batch_size=1,
            seed=seed,
        )
        preds, gts = collect_predictions(model, val_loader, device)
        residuals = preds - gts
        all_residuals.extend(residuals)
        all_preds.extend(preds)
        all_gts.extend(gts)
        print(f"  Val n={len(preds)}, MAE={np.abs(residuals).mean():.4f}, mean_residual={residuals.mean():.4f}")

    if not all_residuals:
        print("ERROR: No predictions collected. Check checkpoint and data paths.")
        sys.exit(1)

    residuals = np.array(all_residuals)
    preds = np.array(all_preds)
    gts = np.array(all_gts)

    # Wilcoxon signed-rank test: H0: median(residual) = 0
    stat, p_value = stats.wilcoxon(residuals)
    print("\n--- Wilcoxon signed-rank test (H0: median residual = 0) ---")
    print(f"  n = {len(residuals)}")
    print(f"  Mean residual (pred - true): {residuals.mean():.4f}")
    print(f"  Median residual: {np.median(residuals):.4f}")
    print(f"  MAE: {np.abs(residuals).mean():.4f}")
    print(f"  Wilcoxon statistic: {stat}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print("\n  --> Significant systematic bias (p < 0.05)")
    else:
        print("\n  --> No significant systematic bias (p > 0.05)")

    # Also report paired t-test for comparison
    t_stat, t_p = stats.ttest_rel(preds, gts)
    print(f"\n  Paired t-test p-value: {t_p:.6f}")


if __name__ == "__main__":
    main()
