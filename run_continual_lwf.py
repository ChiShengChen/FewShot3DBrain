#!/usr/bin/env python3
"""
Run continual learning with LwF (Learning without Forgetting).
Uses feature distillation from previous task model when training sequentially.
"""
import os
import sys
import argparse
import json
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))

from src.data import TASK_CONFIGS, get_few_shot_dataloaders, create_dummy_data
from src.models import TaskModel
from src.train import train_task_cl, evaluate
import torch


def _load_encoder_into_model(model, state_path: str):
    ckpt = torch.load(state_path, map_location="cpu")
    enc_sd = ckpt.get("encoder_state") or ckpt.get("model_state", ckpt)
    if isinstance(enc_sd, dict) and any("_orig_mod" in k for k in enc_sd.keys()):
        enc_sd = {k.replace("_orig_mod.", ""): v for k, v in enc_sd.items()}
    backbone = model.backbone
    if hasattr(backbone, "encoder"):
        model_sd = backbone.encoder.state_dict()
        load_sd = {k: v for k, v in enc_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        backbone.encoder.load_state_dict(load_sd, strict=False)
    else:
        load_sd = {k: v for k, v in enc_sd.items() if k in backbone.state_dict() and backbone.state_dict()[k].shape == v.shape}
        backbone.load_state_dict(load_sd, strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/preprocessed")
    parser.add_argument("--save_dir", type=str, default="./outputs/continual_lwf")
    parser.add_argument("--tasks", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--n_shot", type=int, default=32, choices=[16, 32, 64])
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--distill_weight", type=float, default=0.5)
    parser.add_argument("--create_dummy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = sorted(set(args.tasks))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = (args.patch_size,) * 3

    if args.create_dummy:
        dummy_dir = os.path.join(args.data_dir, "dummy")
        for tid in tasks:
            create_dummy_data(dummy_dir, task_id=tid, n_samples=64)
        args.data_dir = dummy_dir

    adapters_dir = os.path.join(args.save_dir, "adapters")
    os.makedirs(adapters_dir, exist_ok=True)
    prev_encoder_path = None
    old_model = None
    all_metrics = {}

    for task_id in tasks:
        cfg = TASK_CONFIGS[task_id]
        print(f"\n=== Task {task_id}: {cfg['name']} (LwF) ===")

        train_loader, val_loader = get_few_shot_dataloaders(
            args.data_dir, task_id=task_id, n_shot=args.n_shot,
            patch_size=patch_size, batch_size=2, seed=args.seed,
        )

        model = TaskModel(
            task_id=task_id,
            input_channels=cfg["modalities"],
            patch_size=patch_size,
            pretrained_path=args.pretrained_path if task_id == tasks[0] else None,
            model_name="unet_b",
            lora_r=0,
        )
        for p in model.parameters():
            p.requires_grad = True

        if task_id != tasks[0] and prev_encoder_path:
            _load_encoder_into_model(model, prev_encoder_path)

        metrics = train_task_cl(
            model, train_loader, val_loader, task_id, cfg["task_type"],
            device=device, epochs=args.epochs, save_dir=os.path.join(adapters_dir, f"task{task_id}"),
            old_model=old_model, distill_weight=args.distill_weight,
        )
        all_metrics[f"task{task_id}"] = metrics
        print(f"Task {task_id} done: {metrics}")

        old_model = copy.deepcopy(model)
        old_model.eval()
        for p in old_model.parameters():
            p.requires_grad = False

        enc_path = os.path.join(adapters_dir, f"backbone_after_task{task_id}.pt")
        enc_sd = model.backbone.encoder.state_dict() if hasattr(model.backbone, "encoder") else model.backbone.state_dict()
        torch.save({"encoder_state": enc_sd, "task_id": task_id}, enc_path)
        prev_encoder_path = enc_path

    if 2 in tasks and 3 in tasks:
        _, val_loader = get_few_shot_dataloaders(args.data_dir, task_id=2, n_shot=args.n_shot, patch_size=patch_size, batch_size=2, seed=args.seed)
        cfg = TASK_CONFIGS[2]
        model = TaskModel(task_id=2, input_channels=cfg["modalities"], patch_size=patch_size, pretrained_path=None, lora_r=0)
        ckpt = torch.load(os.path.join(adapters_dir, "task2", "task2_best.pt"), map_location="cpu")
        model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
        enc = torch.load(os.path.join(adapters_dir, "backbone_after_task3.pt"), map_location="cpu")
        model.backbone.encoder.load_state_dict(enc.get("encoder_state", enc), strict=False)
        all_metrics["task2_after_t3"] = evaluate(model, val_loader, 2, "segmentation", device)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved to {args.save_dir}/metrics.json")


if __name__ == "__main__":
    main()
