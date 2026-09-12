#!/usr/bin/env python3
"""
Corrected benchmark driver for few-shot continual learning on 3D brain MRI.

Differences from the original run_continual_*.py scripts, all of which changed the
numbers rather than just the plumbing:

  1. Scalar labels reach the model. The crop step used to discard them, so every
     regression run trained against a constant age of 50.0.
  2. Three-way subject-grouped splits. Epoch selection happens on val; the reported
     number comes from a test set no checkpoint ever saw, with no patient shared
     across splits.
  3. Whole-volume deterministic scoring. Segmentation uses sliding-window inference
     with per-case Dice and HD95; scalar tasks average a fixed patch grid.
  4. Every task in the sequence starts from the pretrained backbone. Previously only
     the first task loaded the checkpoint and later tasks adapted random weights.

Usage:
  python scripts/run_benchmark_v2.py --method lora --tasks 2 3 --n_shot 32 --seed 42
  python scripts/run_benchmark_v2.py --method all --tasks 2 3 --n_shot 32 --seed 42
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))

from src.data import TASK_CONFIGS, FewShotFOMODataset, _seg_collate  # noqa: E402
from src.eval import evaluate_split  # noqa: E402
from src.models import TaskModel  # noqa: E402
from src.splits import load_or_make_splits  # noqa: E402
from src.train import train_task  # noqa: E402

METHODS = ["lora", "lora_head", "lora_encoder_only", "sequential_linear", "sequential_ft",
           "pertask_lora"]


def make_loaders(files_train, files_val, task_type, patch_size, batch_size):
    collate = _seg_collate if task_type == "segmentation" else None
    train_ds = FewShotFOMODataset(files_train, patch_size, task_type)
    val_ds = FewShotFOMODataset(files_val, patch_size, task_type, deterministic=True)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                   pin_memory=True, collate_fn=collate),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
                   collate_fn=collate),
    )


def build_model(method, task_id, cfg, patch_size, pretrained, lora_r):
    """Every method starts from the same pretrained backbone; they differ in what is trainable.

    Trainable budgets differ by more than two orders of magnitude between these methods
    (LoRA ~47k on segmentation vs. Sequential Linear's whole decoder at ~8.4M), so the
    parameter counts are reported alongside every result rather than left implicit.
    """
    if method in ("lora", "lora_head", "pertask_lora"):
        model = TaskModel(task_id, cfg["modalities"], patch_size, pretrained_path=pretrained,
                          model_name="unet_b", lora_r=lora_r, lora_decoder=True)
    elif method == "lora_encoder_only":
        model = TaskModel(task_id, cfg["modalities"], patch_size, pretrained_path=pretrained,
                          model_name="unet_b", lora_r=lora_r, lora_decoder=False)
    else:
        model = TaskModel(task_id, cfg["modalities"], patch_size, pretrained_path=pretrained,
                          model_name="unet_b", lora_r=0, lora_decoder=False)

    if method == "sequential_ft":
        for p in model.parameters():
            p.requires_grad = True
    elif method == "sequential_linear":
        # Original baseline semantics: freeze the encoder, train everything downstream
        # (for segmentation that is the entire decoder).
        for n, p in model.named_parameters():
            p.requires_grad = "encoder" not in n
    elif method == "lora_head":
        # Segmentation under plain LoRA leaves the output projection frozen at its
        # pretrained value, so the model can never re-map channels onto the new label
        # space. Unfreezing it costs 66 parameters.
        for n, p in model.named_parameters():
            if "out_conv" in n:
                p.requires_grad = True
    return model


def trainable_report(model):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return {"trainable": int(tr), "total": int(tot), "trainable_pct": 100.0 * tr / max(tot, 1)}


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = (args.patch_size,) * 3
    # Rank is part of the configuration identity for LoRA methods; r=8 is the default and
    # keeps the un-suffixed directory so existing results stay where they are.
    run_name = args.method
    if args.method.startswith("lora") and args.lora_r != 8:
        run_name = f"{args.method}_r{args.lora_r}"
    task_tag = "" if list(args.tasks) == [2, 3] else "_t" + "".join(str(t) for t in args.tasks)
    out_dir = Path(args.save_dir) / f"n{args.n_shot}" / (run_name + task_tag) / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"method": args.method, "seed": args.seed, "n_shot": args.n_shot,
               "tasks": args.tasks,
               "eval_protocol": {"overlap": args.overlap, "patch": list(patch_size),
                                 "eval_batch_size": args.eval_batch_size,
                                 "max_test_subjects": args.max_test_subjects},
               "per_task": {}}
    shared_model = None

    for pos, task_id in enumerate(args.tasks):
        cfg = TASK_CONFIGS[task_id]
        splits = load_or_make_splits(args.data_dir, task_id, args.n_shot, seed=args.seed,
                                     split_dir=args.split_dir,
                                     max_test_subjects=args.max_test_subjects)
        print(f"\n=== [{args.method}] task {task_id} ({cfg['task_type']}) "
              f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} ===",
              flush=True)

        train_loader, val_loader = make_loaders(splits["train"], splits["val"],
                                                cfg["task_type"], patch_size, args.batch_size)

        # sequential_* carry the backbone forward through the sequence (that is what makes
        # them forget); the isolated methods rebuild from the pretrained checkpoint.
        model = build_model(args.method, task_id, cfg, patch_size, args.pretrained_path, args.lora_r)
        if args.method in ("sequential_linear", "sequential_ft") and shared_model is not None:
            model.load_state_dict(_carry_backbone(shared_model, model), strict=False)

        params = trainable_report(model)
        print(f"trainable params: {params['trainable']:,} "
              f"({params['trainable_pct']:.3f}% of {params['total']:,})", flush=True)

        t0 = time.perf_counter()
        val_metrics = train_task(model, train_loader, val_loader, task_id=task_id,
                                 task_type=cfg["task_type"], device=device,
                                 epochs=args.epochs, lr=args.lr,
                                 save_dir=str(out_dir / f"task{task_id}"))
        train_sec = time.perf_counter() - t0

        # Restore the epoch val picked, then score the untouched test split.
        ckpt = out_dir / f"task{task_id}" / f"task{task_id}_best.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state"])
        model = model.to(device)

        # Training is the expensive part and the checkpoint is already on disk, so an
        # evaluation failure is recorded rather than allowed to discard the run.
        try:
            test_metrics = evaluate_split(model, splits["test"], task_id, device,
                                          patch=patch_size, overlap=args.overlap,
                                          batch_size=args.eval_batch_size,
                                          max_channels=cfg["modalities"],
                                          return_per_case=True)
            per_case = test_metrics.pop("per_case", None)
            if per_case is not None:
                (out_dir / f"task{task_id}_per_case.json").write_text(json.dumps(per_case, indent=2))
        except Exception as exc:  # noqa: BLE001 - the queue must keep moving
            import traceback
            traceback.print_exc()
            test_metrics = {"error": f"{type(exc).__name__}: {exc}"}

        results["per_task"][f"task{task_id}"] = {
            "position": pos,
            "val_selected": val_metrics,
            "test": test_metrics,
            "params": params,
            "train_sec": train_sec,
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
        }
        print(f"task {task_id} TEST: {test_metrics}", flush=True)
        (out_dir / "results.partial.json").write_text(json.dumps(results, indent=2))

        if args.method in ("sequential_linear", "sequential_ft"):
            shared_model = model

        # Backward transfer: re-score every earlier task with the model as it now stands.
        if args.method in ("sequential_linear", "sequential_ft") and pos > 0:
            for prev in args.tasks[:pos]:
                pcfg = TASK_CONFIGS[prev]
                psplits = load_or_make_splits(args.data_dir, prev, args.n_shot, seed=args.seed,
                                              split_dir=args.split_dir,
                                              max_test_subjects=args.max_test_subjects)
                prev_model = build_model(args.method, prev, pcfg, patch_size,
                                         args.pretrained_path, args.lora_r)
                # Restore task `prev` exactly as it was left, then overwrite only the
                # shared encoder with its current (post-task-`task_id`) state. That is
                # what forgetting means here: the decoder/head of the old task is intact,
                # but the representation underneath it has moved.
                prev_ckpt = out_dir / f"task{prev}" / f"task{prev}_best.pt"
                if prev_ckpt.exists():
                    prev_model.load_state_dict(
                        torch.load(prev_ckpt, map_location="cpu")["model_state"], strict=False)
                prev_model.load_state_dict(_carry_encoder(model, prev_model), strict=False)
                prev_model = prev_model.to(device)
                try:
                    after = evaluate_split(prev_model, psplits["test"], prev, device,
                                           patch=patch_size, overlap=args.overlap,
                                           batch_size=args.eval_batch_size,
                                           max_channels=pcfg["modalities"])
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    after = {"error": f"{type(exc).__name__}: {exc}"}
                results["per_task"][f"task{prev}"][f"test_after_task{task_id}"] = after
                print(f"task {prev} re-scored after task {task_id}: {after}", flush=True)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir / 'results.json'}")
    return results


def _carry_backbone(src_model, dst_model):
    """Copy shared backbone weights across a task switch, leaving task heads alone."""
    src = src_model.state_dict()
    dst = dst_model.state_dict()
    out = {}
    for k, v in dst.items():
        if k in src and src[k].shape == v.shape and k.startswith("backbone"):
            out[k] = src[k]
    return out


def _carry_encoder(src_model, dst_model):
    """Copy only the shared encoder, leaving the old task's decoder/head untouched."""
    src = src_model.state_dict()
    dst = dst_model.state_dict()
    return {k: src[k] for k, v in dst.items()
            if k.startswith("backbone.encoder") and k in src and src[k].shape == v.shape}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="lora", choices=METHODS + ["all"])
    ap.add_argument("--tasks", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--data_dir", default=str(ROOT / "data/preprocessed"))
    ap.add_argument("--split_dir", default=str(ROOT / "data/splits_v3"))
    ap.add_argument("--save_dir", default=str(ROOT / "outputs/benchmark_v2"))
    ap.add_argument("--pretrained_path", default=str(ROOT / "weights/fomo25_mmunetvae_pretrained.ckpt"))
    ap.add_argument("--n_shot", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--patch_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--max_test_subjects", type=int, default=250)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    methods = METHODS if args.method == "all" else [args.method]
    for m in methods:
        args.method = m
        run(args)


if __name__ == "__main__":
    main()
