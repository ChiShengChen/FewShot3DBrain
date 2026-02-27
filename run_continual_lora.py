#!/usr/bin/env python3
"""
Run few-shot continual learning with LoRA adapters.
Usage:
  python run_continual_lora.py --data_dir ./data/preprocessed --n_shot 32 [--pretrained_path ...]
"""
import os
import sys
import argparse
from pathlib import Path

# Project layout
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))

from src.data import (
    TASK_CONFIGS,
    get_few_shot_dataloaders,
    create_dummy_data,
)

from src.models import TaskModel
from src.train import train_task, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/preprocessed")
    parser.add_argument("--save_dir", type=str, default="./outputs/continual_lora")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--n_shot", type=int, default=32, choices=[16, 32, 64])
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_decoder", action="store_true",
                        help="Add LoRA to T2 decoder (default: encoder-only for segmentation)")
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3],
                        help="Tasks to run, e.g. --tasks 2 3 to skip Task 1")
    parser.add_argument("--create_dummy", action="store_true", help="Create dummy data if no data found")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = list(dict.fromkeys(args.tasks))  # preserve order, no duplicates
    for t in tasks:
        if t not in (1, 2, 3):
            print(f"Error: invalid task {t}. Use 1, 2, or 3.")
            sys.exit(1)

    import torch
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Tasks: {tasks}")
    print(f"LoRA: encoder-only" if not args.lora_decoder else "LoRA: encoder+decoder")

    patch_size = (args.patch_size,) * 3

    # Check for data; create dummy if requested
    if args.create_dummy:
        dummy_dir = os.path.join(args.data_dir, "dummy")
        for tid in tasks:
            create_dummy_data(dummy_dir, task_id=tid, n_samples=64)
        args.data_dir = dummy_dir

    all_metrics = {}
    adapters_dir = os.path.join(args.save_dir, "adapters")

    for task_id in tasks:
        cfg = TASK_CONFIGS[task_id]
        print(f"\n=== Task {task_id}: {cfg['name']} ({cfg['task_type']}) ===")

        try:
            train_loader, val_loader = get_few_shot_dataloaders(
                args.data_dir,
                task_id=task_id,
                n_shot=args.n_shot,
                patch_size=patch_size,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Run preprocessing or use --create_dummy to generate dummy data.")
            sys.exit(1)

        # Build model: frozen backbone + LoRA for this task
        # First task in sequence gets pretrained weights
        use_pretrained = (task_id == tasks[0]) and args.pretrained_path
        model = TaskModel(
            task_id=task_id,
            input_channels=cfg["modalities"],
            patch_size=patch_size,
            pretrained_path=args.pretrained_path if use_pretrained else None,
            model_name="unet_b",
            lora_r=args.lora_r,
            lora_decoder=args.lora_decoder,
        )

        task_save = os.path.join(adapters_dir, f"task{task_id}")
        metrics = train_task(
            model,
            train_loader,
            val_loader,
            task_id=task_id,
            task_type=cfg["task_type"],
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            save_dir=task_save,
        )
        all_metrics[f"task{task_id}"] = metrics
        print(f"Task {task_id} done: {metrics}")

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        import json
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {args.save_dir}/metrics.json")


if __name__ == "__main__":
    main()
