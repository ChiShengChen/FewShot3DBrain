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
from src.swin_model import SwinTaskModel  # noqa: E402

BACKBONES = {"unet_b": TaskModel, "swinunetr": SwinTaskModel}
DEFAULT_WEIGHTS = {"unet_b": str(ROOT / "weights/fomo25_mmunetvae_pretrained.ckpt"),
                   "swinunetr": str(ROOT / "weights/model_swinvit.pt")}
from src.splits import load_or_make_splits  # noqa: E402
from src.replay_buffer import ReplayBuffer  # noqa: E402
from src.train import compute_fisher_diagonal, train_task, train_task_cl  # noqa: E402

METHODS = ["lora", "lora_head", "lora_encoder_only", "sequential_linear", "sequential_ft",
           "ewc", "lwf", "replay", "bitfit", "pertask_lora", "adapter"]
# Methods that carry one model through the task sequence (and can therefore forget).
# The regularisation / distillation / replay baselines are full fine-tuning plus a
# forgetting-mitigation term, so they belong here too.
SEQUENTIAL = ("sequential_linear", "sequential_ft", "ewc", "lwf", "replay")
CL_METHODS = ("ewc", "lwf", "replay")


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


def build_model(method, task_id, cfg, patch_size, pretrained, lora_r, adapter_dim=16, backbone="unet_b"):
    """Every method starts from the same pretrained backbone; they differ in what is trainable.

    Trainable budgets differ by more than two orders of magnitude between these methods
    (LoRA ~47k on segmentation vs. Sequential Linear's whole decoder at ~8.4M), so the
    parameter counts are reported alongside every result rather than left implicit.
    """
    Model = BACKBONES[backbone]
    if method in ("lora", "lora_head", "pertask_lora"):
        model = Model(task_id, cfg["modalities"], patch_size, pretrained_path=pretrained,
                          model_name=backbone, lora_r=lora_r, lora_decoder=True)
    elif method == "adapter":
        # Houlsby bottleneck adapters at a budget matched to LoRA r=16 (dim=16: 93.7K vs
        # 93.8K on segmentation). Task-isolated like LoRA; output projection trained below.
        model = Model(task_id, cfg["modalities"], patch_size, pretrained_path=pretrained,
                          model_name=backbone, lora_r=0, lora_decoder=True, adapter_dim=adapter_dim)
    elif method == "lora_encoder_only":
        model = Model(task_id, cfg["modalities"], patch_size, pretrained_path=pretrained,
                          model_name=backbone, lora_r=lora_r, lora_decoder=False)
    else:
        model = Model(task_id, cfg["modalities"], patch_size, pretrained_path=pretrained,
                          model_name=backbone, lora_r=0, lora_decoder=False)

    if method in ("sequential_ft",) + CL_METHODS:
        for p in model.parameters():
            p.requires_grad = True
    elif method == "sequential_linear":
        # Original baseline semantics: freeze the encoder, train everything downstream
        # (for segmentation that is the entire decoder).
        for n, p in model.named_parameters():
            p.requires_grad = "encoder" not in n
    elif method in ("lora_head", "adapter"):
        # Segmentation under plain LoRA leaves the output projection frozen at its
        # pretrained value, so the model can never re-map channels onto the new label
        # space. Unfreezing it costs 66 parameters. The adapter comparator gets the same.
        for n, p in model.named_parameters():
            if "out_conv" in n:
                p.requires_grad = True
    elif method == "bitfit":
        # PEFT comparator at a similar budget to LoRA: bias terms only (plus the task
        # head / output projection, as every PEFT recipe trains the new head). Built with
        # lora_r=0, so nothing else is trainable. Task-isolated like LoRA: a fresh copy
        # of the backbone per task, so BWT is zero by construction.
        for n, p in model.named_parameters():
            p.requires_grad = n.endswith(".bias") or "out_conv" in n or n.startswith("head")
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
    if args.method == "adapter" and args.adapter_dim != 16:
        run_name = f"adapter_d{args.adapter_dim}"
    # A second backbone files its runs under a prefix so the U-Net tables never pick them up.
    if args.backbone != "unet_b":
        run_name = {"swinunetr": "swin"}[args.backbone] + "_" + run_name
    task_tag = "" if list(args.tasks) == [2, 3] else "_t" + "".join(str(t) for t in args.tasks)
    # Free-form tag for controls that change something the name cannot express (e.g. a
    # randomly initialised backbone), so they never overwrite the main configuration.
    extra_tag = f"_{args.tag}" if args.tag else ""
    # --full_train keeps the n-shot split's val/test and trains on every other subject:
    # the "how far is 32-shot from the ceiling" reference, filed under nfull/.
    n_tag = "full" if args.full_train else str(args.n_shot)
    out_dir = Path(args.save_dir) / f"n{n_tag}" / (run_name + task_tag + extra_tag) / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"method": args.method, "seed": args.seed, "n_shot": args.n_shot,
               "tasks": args.tasks,
               "eval_protocol": {"overlap": args.overlap, "patch": list(patch_size),
                                 "eval_batch_size": args.eval_batch_size,
                                 "max_test_subjects": args.max_test_subjects},
               "per_task": {}}
    shared_model = None
    # State the forgetting-mitigation baselines carry between tasks.
    cl_state = {
        "fisher": None, "optimal": None,          # EWC: diagonal Fisher of the previous task
        "old_model": None,                        # LwF / replay: frozen copy of the previous model
        "buffer": ReplayBuffer(capacity=args.replay_capacity, seed=args.seed)
                  if args.method == "replay" else None,
        "seen": [],                               # task ids already learned (replay sources)
    }

    for pos, task_id in enumerate(args.tasks):
        cfg = TASK_CONFIGS[task_id]
        splits = load_or_make_splits(args.data_dir, task_id, args.n_shot, seed=args.seed,
                                     split_dir=args.split_dir,
                                     max_test_subjects=args.max_test_subjects)
        if args.full_train:
            # Same held-out val/test as the few-shot run of this seed (so the number is
            # directly comparable); train = every usable case that is in neither.
            from src.data import list_case_files
            # Split files may hold relative paths (older seeds) or absolute ones; match on
            # the case name, which is unique within a task.
            held = {os.path.basename(f) for f in splits["val"]} | {os.path.basename(f) for f in splits["test"]}
            splits = dict(splits)
            splits["train"] = [f for f in list_case_files(args.data_dir, task_id)
                               if os.path.basename(f) not in held]
            if not splits["train"]:
                raise SystemExit(f"task {task_id}: no cases outside the n{args.n_shot} val/test split; "
                                 "the few-shot protocol already uses every other subject")
        print(f"\n=== [{args.method}] task {task_id} ({cfg['task_type']}) "
              f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} ===",
              flush=True)

        train_loader, val_loader = make_loaders(splits["train"], splits["val"],
                                                cfg["task_type"], patch_size, args.batch_size)

        # sequential_* carry the backbone forward through the sequence (that is what makes
        # them forget); the isolated methods rebuild from the pretrained checkpoint.
        model = build_model(args.method, task_id, cfg, patch_size, args.pretrained_path, args.lora_r,
                            adapter_dim=args.adapter_dim, backbone=args.backbone)
        if args.method in SEQUENTIAL and shared_model is not None:
            model.load_state_dict(_carry_backbone(shared_model, model), strict=False)

        params = trainable_report(model)
        print(f"trainable params: {params['trainable']:,} "
              f"({params['trainable_pct']:.3f}% of {params['total']:,})", flush=True)

        t0 = time.perf_counter()
        common = dict(task_id=task_id, task_type=cfg["task_type"], device=device,
                      epochs=args.epochs, lr=args.lr, save_dir=str(out_dir / f"task{task_id}"))
        if args.method in CL_METHODS:
            val_metrics = train_task_cl(
                model, train_loader, val_loader, **common,
                ewc_fisher=cl_state["fisher"] if args.method == "ewc" else None,
                ewc_optimal=cl_state["optimal"] if args.method == "ewc" else None,
                lambda_ewc=args.lambda_ewc,
                old_model=cl_state["old_model"] if args.method in ("lwf", "replay") else None,
                distill_weight=args.distill_weight,
                replay_buffer=cl_state["buffer"],
                replay_task_ids=list(cl_state["seen"]) if args.method == "replay" else None,
                replay_weight=args.distill_weight,
            )
        else:
            val_metrics = train_task(model, train_loader, val_loader, **common)
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

        if args.method in SEQUENTIAL:
            shared_model = model

        # Capture what the mitigation baselines need from this task before moving on.
        # Done on the checkpoint that was kept (best epoch), which is what `model` now holds.
        if args.method == "ewc":
            cl_state["fisher"], cl_state["optimal"] = compute_fisher_diagonal(
                model, train_loader, task_id, cfg["task_type"], device)
        if args.method in ("lwf", "replay"):
            import copy
            cl_state["old_model"] = copy.deepcopy(model).eval()
            for p in cl_state["old_model"].parameters():
                p.requires_grad = False
        if args.method == "replay":
            cl_state["buffer"].add_from_loader(task_id, train_loader, max_samples=args.replay_capacity)
            cl_state["seen"].append(task_id)

        # Backward transfer: re-score every earlier task with the model as it now stands.
        if args.method in SEQUENTIAL and pos > 0:
            for prev in args.tasks[:pos]:
                pcfg = TASK_CONFIGS[prev]
                psplits = load_or_make_splits(args.data_dir, prev, args.n_shot, seed=args.seed,
                                              split_dir=args.split_dir,
                                              max_test_subjects=args.max_test_subjects)
                prev_model = build_model(args.method, prev, pcfg, patch_size,
                                         args.pretrained_path, args.lora_r, backbone=args.backbone)
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
    ap.add_argument("--backbone", default="unet_b", choices=list(BACKBONES),
                    help="unet_b: FOMO 3D U-Net (main study); swinunetr: MONAI Swin UNETR with the "
                         "self-supervised encoder of Tang et al. (backbone-independence check)")
    ap.add_argument("--pretrained_path", default=None,
                    help="checkpoint for the backbone (default: the backbone's own); '' = random init")
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
    ap.add_argument("--tag", default="", help="suffix for the run directory (controls/ablations)")
    ap.add_argument("--adapter_dim", type=int, default=16,
                    help="bottleneck width for --method adapter (16 matches LoRA r=16's budget)")
    ap.add_argument("--full_train", action="store_true",
                    help="train on every subject outside the n-shot split's val/test (ceiling reference; "
                         "results go under nfull/)")
    # Forgetting-mitigation baselines (defaults follow the original run_continual_*.py).
    ap.add_argument("--lambda_ewc", type=float, default=1000.0)
    ap.add_argument("--distill_weight", type=float, default=0.5,
                    help="LwF feature-distillation weight; also used as the replay weight")
    ap.add_argument("--replay_capacity", type=int, default=100)
    args = ap.parse_args()
    if args.pretrained_path is None:
        args.pretrained_path = DEFAULT_WEIGHTS[args.backbone]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    methods = METHODS if args.method == "all" else [args.method]
    for m in methods:
        args.method = m
        run(args)


if __name__ == "__main__":
    main()
