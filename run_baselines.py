#!/usr/bin/env python3
"""
Run baseline methods for continual learning comparison.
- Linear probe: freeze backbone, train only heads
- Sequential FT: full fine-tune sequentially (catastrophic forgetting baseline)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))

from src.data import TASK_CONFIGS, get_few_shot_dataloaders, create_dummy_data
from src.train import train_task


def build_linear_probe_model(task_id: int, input_channels: int, patch_size: tuple, pretrained_path=None):
    """Frozen backbone + trainable head/decoder only (no LoRA)."""
    from src.models import TaskModel
    model = TaskModel(
        task_id=task_id,
        input_channels=input_channels,
        patch_size=patch_size,
        pretrained_path=pretrained_path,
        lora_r=0,  # no LoRA
    )
    # Freeze encoder, train head (cls/reg) or decoder (seg)
    for n, p in model.named_parameters():
        if "encoder" in n:
            p.requires_grad = False
        else:
            p.requires_grad = True
    return model


def build_sequential_ft_model(task_id: int, input_channels: int, patch_size: tuple, pretrained_path=None):
    """Full fine-tuning - all params trainable."""
    from src.models import TaskModel
    model = TaskModel(
        task_id=task_id,
        input_channels=input_channels,
        patch_size=patch_size,
        pretrained_path=pretrained_path,
        lora_r=0,
    )
    for p in model.parameters():
        p.requires_grad = True
    return model


def main():
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="linear", choices=["linear", "sequential_ft"])
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3],
                        help="Tasks to run, e.g. --tasks 2 3 to skip Task 1")
    parser.add_argument("--data_dir", type=str, default="./data/preprocessed")
    parser.add_argument("--save_dir", type=str, default="./outputs/baselines")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--n_shot", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--create_dummy", action="store_true")
    args = parser.parse_args()

    tasks = sorted(set(args.tasks))
    for t in tasks:
        if t not in (1, 2, 3):
            print(f"Error: invalid task {t}. Use 1, 2, or 3.")
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = (32,) * 3

    if args.create_dummy:
        d = os.path.join(args.data_dir, "dummy")
        for tid in tasks:
            create_dummy_data(d, tid, 64)
        args.data_dir = d

    results = {}
    for task_id in tasks:
        cfg = TASK_CONFIGS[task_id]
        train_loader, val_loader = get_few_shot_dataloaders(
            args.data_dir, task_id, args.n_shot, patch_size
        )
        if args.baseline == "linear":
            model = build_linear_probe_model(task_id, cfg["modalities"], patch_size, args.pretrained_path)
        else:
            model = build_sequential_ft_model(task_id, cfg["modalities"], patch_size, args.pretrained_path)
        save_dir = os.path.join(args.save_dir, args.baseline, f"task{task_id}")
        m = train_task(model, train_loader, val_loader, task_id, cfg["task_type"], device, epochs=args.epochs, save_dir=save_dir)
        results[f"task{task_id}"] = m
        print(f"{args.baseline} Task {task_id}: {m}")
    print("\nResults:", results)


if __name__ == "__main__":
    main()
