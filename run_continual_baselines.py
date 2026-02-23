#!/usr/bin/env python3
"""
Run continual learning baselines: one model, sequential T1→T2→T3.
- sequential_linear: frozen backbone, train only heads (fair comparison to LoRA)
- sequential_ft: full fine-tune sequentially (catastrophic forgetting baseline)
"""
import os
import sys
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))

from src.data import TASK_CONFIGS, get_few_shot_dataloaders, create_dummy_data
from src.models import TaskModel
from src.train import train_task, evaluate
import torch


def _get_encoder_state(model):
    """Extract encoder state dict (shared backbone)."""
    backbone = model.backbone
    if hasattr(backbone, "encoder"):
        return {k.replace("encoder.", ""): v for k, v in backbone.encoder.state_dict().items()}
    return backbone.state_dict()


def _reeval_t2_after_t3(adapters_dir: str, tasks: list, args, patch_size: tuple, device, all_metrics: dict, baseline: str):
    """
    Re-evaluate T2 after T3 training.
    - sequential_ft: load encoder from backbone_after_task3 (overwritten by T3) → measures forgetting
    - sequential_linear: load encoder from backbone_after_task2 (frozen, same as T2) → should match T2
    """
    _, val_loader = get_few_shot_dataloaders(
        args.data_dir,
        task_id=2,
        n_shot=args.n_shot,
        patch_size=patch_size,
        batch_size=2,
        seed=args.seed,
    )
    cfg = TASK_CONFIGS[2]
    model = TaskModel(task_id=2, input_channels=cfg["modalities"], patch_size=patch_size, pretrained_path=None, lora_r=0)
    model.eval()

    task2_ckpt_path = os.path.join(adapters_dir, "task2", "task2_best.pt")
    if not os.path.exists(task2_ckpt_path):
        print("WARNING: task2_best.pt not found, skipping T2 re-eval.")
        return

    # sequential_ft: encoder overwritten by T3 → use backbone_after_task3 (measures forgetting)
    # sequential_linear: encoder frozen → use backbone_after_task2 (3ch, correct for T2)
    enc_file = "backbone_after_task3.pt" if baseline == "sequential_ft" else "backbone_after_task2.pt"
    backbone_enc_path = os.path.join(adapters_dir, enc_file)
    if not os.path.exists(backbone_enc_path):
        print(f"WARNING: {enc_file} not found, skipping T2 re-eval.")
        return

    ckpt = torch.load(task2_ckpt_path, map_location="cpu")
    t2_sd = ckpt.get("model_state", ckpt)

    enc_ckpt = torch.load(backbone_enc_path, map_location="cpu")
    enc_sd = enc_ckpt.get("encoder_state", enc_ckpt)
    model_enc_sd = model.backbone.encoder.state_dict()
    load_enc = {k: v for k, v in enc_sd.items() if k in model_enc_sd and model_enc_sd[k].shape == v.shape}
    model.backbone.encoder.load_state_dict(load_enc, strict=False)

    dec_sd = {k.replace("backbone.decoder.", ""): v for k, v in t2_sd.items() if k.startswith("backbone.decoder.")}
    if dec_sd:
        model.backbone.decoder.load_state_dict(dec_sd, strict=False)

    model = model.to(device)
    metrics = evaluate(model, val_loader, 2, "segmentation", device)
    all_metrics["task2_after_t3"] = metrics
    t2_orig = all_metrics.get("task2", {}).get("dice", 0)
    delta = float(t2_orig) - metrics["dice"]
    print(f"\n=== T2 re-eval after T3 (catastrophic forgetting) ===")
    print(f"T2 Dice right after T2: {t2_orig:.4f}")
    print(f"T2 Dice after T3 training: {metrics['dice']:.4f}")
    print(f"Forgetting Δ = {delta:.4f}")


def _load_encoder_into_model(model, state_path: str):
    """Load encoder weights from checkpoint into model backbone."""
    ckpt = torch.load(state_path, map_location="cpu")
    enc_sd = ckpt.get("encoder_state") or ckpt.get("model_state", ckpt)
    if isinstance(enc_sd, dict) and any("_orig_mod" in k for k in enc_sd.keys()):
        enc_sd = {k.replace("_orig_mod.", ""): v for k, v in enc_sd.items()}
    backbone = model.backbone
    if hasattr(backbone, "encoder"):
        model_sd = backbone.encoder.state_dict()
        load_sd = {k: v for k, v in enc_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        backbone.encoder.load_state_dict(load_sd, strict=False)
        print(f"Loaded {len(load_sd)}/{len(model_sd)} encoder keys from {state_path}")
    else:
        load_sd = {k: v for k, v in enc_sd.items() if k in backbone.state_dict() and backbone.state_dict()[k].shape == v.shape}
        backbone.load_state_dict(load_sd, strict=False)
        print(f"Loaded {len(load_sd)} keys from {state_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="sequential_linear",
                        choices=["sequential_linear", "sequential_ft"])
    parser.add_argument("--data_dir", type=str, default="./data/preprocessed")
    parser.add_argument("--save_dir", type=str, default="./outputs/continual_baselines")
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3],
                        help="Tasks to run, e.g. --tasks 2 3 to skip Task 1")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--n_shot", type=int, default=32, choices=[16, 32, 64])
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--create_dummy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = sorted(set(args.tasks))
    for t in tasks:
        if t not in (1, 2, 3):
            print(f"Error: invalid task {t}. Use 1, 2, or 3.")
            sys.exit(1)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Baseline: {args.baseline} (sequential {'→'.join(str(t) for t in tasks)})")

    patch_size = (args.patch_size,) * 3
    if args.create_dummy:
        dummy_dir = os.path.join(args.data_dir, "dummy")
        for tid in tasks:
            create_dummy_data(dummy_dir, task_id=tid, n_samples=64)
        args.data_dir = dummy_dir

    adapters_dir = os.path.join(args.save_dir, args.baseline, "adapters")
    os.makedirs(adapters_dir, exist_ok=True)
    prev_encoder_path = None

    all_metrics = {}
    for task_id in tasks:
        cfg = TASK_CONFIGS[task_id]
        print(f"\n=== Task {task_id}: {cfg['name']} ({cfg['task_type']}) ===")

        train_loader, val_loader = get_few_shot_dataloaders(
            args.data_dir,
            task_id=task_id,
            n_shot=args.n_shot,
            patch_size=patch_size,             batch_size=2,
            seed=args.seed,
        )

        pretrained = args.pretrained_path if task_id == tasks[0] else None
        lora_r = 0  # no LoRA for baselines

        model = TaskModel(
            task_id=task_id,
            input_channels=cfg["modalities"],
            patch_size=patch_size,
            pretrained_path=pretrained,
            model_name="unet_b",
            lora_r=lora_r,
        )

        if task_id != tasks[0] and prev_encoder_path:
            _load_encoder_into_model(model, prev_encoder_path)

        if args.baseline == "sequential_linear":
            for n, p in model.named_parameters():
                if "encoder" in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True

        task_save = os.path.join(adapters_dir, f"task{task_id}")
        metrics = train_task(
            model, train_loader, val_loader, task_id, cfg["task_type"],
            device=device, epochs=args.epochs, save_dir=task_save,
        )
        all_metrics[f"task{task_id}"] = metrics
        print(f"Task {task_id} done: {metrics}")

        encoder_path = os.path.join(adapters_dir, f"backbone_after_task{task_id}.pt")
        enc_sd = model.backbone.encoder.state_dict() if hasattr(model.backbone, "encoder") else model.backbone.state_dict()
        torch.save({"encoder_state": enc_sd, "task_id": task_id}, encoder_path)
        prev_encoder_path = encoder_path

    # Re-eval T2 after T3 (for BWT: R_3,2) when tasks include 2,3
    if 2 in tasks and 3 in tasks:
        _reeval_t2_after_t3(
            adapters_dir=adapters_dir,
            tasks=tasks,
            args=args,
            patch_size=patch_size,
            device=device,
            all_metrics=all_metrics,
            baseline=args.baseline,
        )

    os.makedirs(os.path.join(args.save_dir, args.baseline), exist_ok=True)
    out_path = os.path.join(args.save_dir, args.baseline, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to {out_path}")


if __name__ == "__main__":
    main()
