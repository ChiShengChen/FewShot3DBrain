#!/usr/bin/env python3
"""
MICCAI 2026 experiments: run EWC, LwF, Replay baselines + n_shot=64 + task order.
Usage:
  python scripts/run_miccai_experiments.py [--phase 1|2|3|all] [--create_dummy] [--seeds 42 43 44]
"""
import json
import os
import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

METHODS = {
    "lora": ("run_continual_lora.py", ["--lora_decoder"]),
    "sequential_linear": ("run_continual_baselines.py", ["--baseline", "sequential_linear"]),
    "sequential_ft": ("run_continual_baselines.py", ["--baseline", "sequential_ft"]),
    "ewc": ("run_continual_ewc.py", []),
    "lwf": ("run_continual_lwf.py", []),
    "replay": ("run_continual_replay.py", []),
}


def metrics_path_for(save_dir: str, method: str) -> Path:
    """Path to metrics.json for a given method."""
    if method in ("sequential_linear", "sequential_ft"):
        return Path(save_dir) / method / "metrics.json"
    return Path(save_dir) / "metrics.json"


def _legacy_metrics_path(method: str, seed: int) -> Path:
    """Path used by run_multi_seed (paper's original runs)."""
    base = ROOT / "outputs" / "multi_seed" / method / f"seed{seed}"
    if method in ("sequential_linear", "sequential_ft"):
        return base / method / "metrics.json"
    return base / "metrics.json"


def _metrics_complete(mp: Path, tasks: list, method: str) -> bool:
    """Check if metrics.json exists and has all required keys (complete run).
    LoRA does not compute task2_after_t3; EWC/LwF/Replay/sequential_* do."""
    if not mp.exists():
        return False
    try:
        with open(mp) as f:
            m = json.load(f)
        task_ids = [int(t) for t in tasks]
        required = [f"task{t}" for t in task_ids]
        if 2 in task_ids and 3 in task_ids and method != "lora":
            required.append("task2_after_t3" if task_ids == [2, 3] else "task3_after_t2")
        return all(k in m for k in required)
    except Exception:
        return False


def run_cmd(cmd: list, cwd=None):
    cwd = cwd or str(ROOT)
    ret = subprocess.run(cmd, cwd=cwd)
    return ret.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3],
                    help="1=n_shot32+baselines, 2=n_shot64, 3=task_order T3→T2")
    ap.add_argument("--data_dir", default="./data/preprocessed")
    ap.add_argument("--pretrained_path", default="./weights/fomo25_mmunetvae_pretrained.ckpt")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--create_dummy", action="store_true")
    ap.add_argument("--skip_existing", action="store_true", default=True,
                    help="Skip if metrics.json exists (default: True)")
    ap.add_argument("--skip_legacy", action="store_true", default=True,
                    help="Also skip if outputs/multi_seed/ has results (paper's original runs)")
    ap.add_argument("--force", action="store_true", help="Re-run even if results exist")
    args = ap.parse_args()

    pret_args = []
    if os.path.isfile(args.pretrained_path):
        pret_args = ["--pretrained_path", args.pretrained_path]

    common = ["--data_dir", args.data_dir, "--epochs", str(args.epochs)]
    if args.create_dummy:
        common.append("--create_dummy")

    if args.phase == 1:
        methods = ["lora", "sequential_linear", "sequential_ft", "ewc", "lwf", "replay"]
        out_base = "outputs/miccai_experiments/n32"
        tasks = ["2", "3"]
        n_shot = "32"
    elif args.phase == 2:
        methods = ["lora", "sequential_linear", "sequential_ft", "ewc", "lwf"]
        out_base = "outputs/miccai_experiments/n64"
        tasks = ["2", "3"]
        n_shot = "64"
    else:  # phase 3: T3→T2 order (t32_t3t2 to avoid overwriting old t32)
        methods = ["lora", "sequential_linear", "sequential_ft"]
        out_base = "outputs/miccai_experiments/t32_t3t2"
        tasks = ["3", "2"]
        n_shot = "32"

    for method in methods:
        for seed in args.seeds:
            save_dir = f"{out_base}/{method}/seed{seed}"
            mp = metrics_path_for(save_dir, method)
            legacy_mp = _legacy_metrics_path(method, seed) if args.skip_legacy else None
            complete = _metrics_complete(mp, tasks, method) or (legacy_mp and _metrics_complete(legacy_mp, tasks, method))
            if not args.force and args.skip_existing and complete:
                print(f"\n>>> {method} seed={seed} [skip: complete]")
                continue
            script, extra = METHODS[method]
            cmd = [sys.executable, script, "--tasks", *tasks, "--n_shot", n_shot, "--seed", str(seed),
                   "--save_dir", save_dir, *common, *pret_args, *extra]
            print(f"\n>>> {method} seed={seed}")
            run_cmd(cmd)
    print(f"\nDone. Results in {out_base}/")


if __name__ == "__main__":
    main()
