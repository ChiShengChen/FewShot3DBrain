#!/usr/bin/env python3
"""
Train a single task with LoRA (for debugging / single-task baseline).
Usage:
  python run_single_task.py --task_id 1 --data_dir ./data --n_shot 32 --create_dummy
"""
import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # our src package (must be first)
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))  # FOMO models, utils (do NOT add fomo_baseline/ to avoid src shadowing)

from src.data import TASK_CONFIGS, get_few_shot_dataloaders, create_dummy_data
from src.models import TaskModel
from src.train import train_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--data_dir", type=str, default="./data/preprocessed")
    parser.add_argument("--save_dir", type=str, default="./outputs/single_task")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--n_shot", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--create_dummy", action="store_true")
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.create_dummy:
        d = os.path.join(args.data_dir, "dummy")
        create_dummy_data(d, task_id=args.task_id, n_samples=64)
        args.data_dir = d

    cfg = TASK_CONFIGS[args.task_id]
    train_loader, val_loader = get_few_shot_dataloaders(
        args.data_dir, args.task_id, args.n_shot, (args.patch_size,) * 3, args.batch_size
    )

    model = TaskModel(
        task_id=args.task_id,
        input_channels=cfg["modalities"],
        patch_size=(args.patch_size,) * 3,
        pretrained_path=args.pretrained_path,
    )

    save_dir = os.path.join(args.save_dir, f"task{args.task_id}")
    metrics = train_task(model, train_loader, val_loader, args.task_id, cfg["task_type"], device, epochs=args.epochs, save_dir=save_dir)
    print("Final:", metrics)


if __name__ == "__main__":
    main()
