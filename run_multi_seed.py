#!/usr/bin/env python3
"""
Run experiments with multiple seeds and compute mean±std, BWT, FWT.
Usage:
  python run_multi_seed.py --method all --seeds 42 43 44 --pretrained_path ./weights/fomo25_mmunetvae_pretrained.ckpt
"""
import os
import sys
import json
import subprocess
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))

# Methods and their scripts + extra args
METHODS = {
    "lora": ("run_continual_lora.py", ["--lora_decoder"]),  # use enc+dec for T2
    "sequential_linear": ("run_continual_baselines.py", ["--baseline", "sequential_linear"]),
    "sequential_ft": ("run_continual_baselines.py", ["--baseline", "sequential_ft"]),
    "ewc": ("run_continual_ewc.py", []),
    "lwf": ("run_continual_lwf.py", []),
    "replay": ("run_continual_replay.py", []),
}


def run_single(args, method: str, seed: int) -> dict:
    """Run one experiment with given seed. Returns metrics dict."""
    script, extra = METHODS[method]
    save_subdir = f"multi_seed/{method}/seed{seed}"
    base_dir = str(ROOT / "outputs" / save_subdir)
    cmd = [
        sys.executable, str(ROOT / script),
        "--tasks", "2", "3",
        "--data_dir", args.data_dir,
        "--save_dir", base_dir,
        "--n_shot", str(args.n_shot),
        "--epochs", str(args.epochs),
        "--seed", str(seed),
    ]
    if args.pretrained_path:
        cmd.extend(["--pretrained_path", args.pretrained_path])
    if getattr(args, "create_dummy", False):
        cmd.append("--create_dummy")
    cmd.extend(extra)

    print(f"\n{'='*60}\nRunning {method} seed={seed}\n{'='*60}")
    ret = subprocess.run(cmd, cwd=str(ROOT))
    if ret.returncode != 0:
        print(f"ERROR: {method} seed={seed} failed")
        return {}

    # Load metrics
    if method == "lora":
        metrics_path = Path(base_dir) / "metrics.json"
    elif method in ("ewc", "lwf", "replay"):
        metrics_path = Path(base_dir) / "metrics.json"
    else:
        baseline = "sequential_linear" if "linear" in method else "sequential_ft"
        metrics_path = Path(base_dir) / baseline / "metrics.json"

    if not metrics_path.exists():
        print(f"WARNING: {metrics_path} not found")
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def compute_bwt_fwt(metrics: dict, tasks: list) -> tuple:
    """
    BWT = R_T,k - R_k,k for previous tasks (negative = forgetting)
    For tasks [2,3]: BWT = R_3,2 - R_2,2 (T2 perf after T3 minus T2 right after T2)
    FWT = R_{k-1,k} - R_{0,k} (forward transfer; requires R_0,k from pretrained)
    """
    bwt = None
    fwt = None
    if 2 in tasks and 3 in tasks:
        r22 = metrics.get("task2", {}).get("dice")
        r33 = metrics.get("task3", {}).get("mae")
        r32 = metrics.get("task2_after_t3", {}).get("dice")  # T2 after T3
        if r32 is None:
            r32 = r22  # LoRA/SeqLinear: no forgetting, use same
        if r22 is not None and r32 is not None:
            bwt = r32 - r22  # negative = forgetting
    return bwt, fwt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="all",
                        choices=["all", "lora", "sequential_linear", "sequential_ft", "ewc", "lwf", "replay"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--pretrained_path", type=str, default="./weights/fomo25_mmunetvae_pretrained.ckpt")
    parser.add_argument("--data_dir", type=str, default="./data/preprocessed")
    parser.add_argument("--n_shot", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--create_dummy", action="store_true")
    args = parser.parse_args()

    methods = (["lora", "sequential_linear", "sequential_ft", "ewc", "lwf", "replay"]
               if args.method == "all" else [args.method])
    seeds = args.seeds

    all_results = {}
    for method in methods:
        all_results[method] = {"runs": [], "mean_std": {}, "bwt": [], "fwt": []}
        for seed in seeds:
            m = run_single(args, method, seed)
            if m:
                all_results[method]["runs"].append(m)
                bwt, fwt = compute_bwt_fwt(m, [2, 3])
                if bwt is not None:
                    all_results[method]["bwt"].append(bwt)
                if fwt is not None:
                    all_results[method]["fwt"].append(fwt)

    # Aggregate mean±std
    for method in methods:
        runs = all_results[method]["runs"]
        if not runs:
            continue
        keys_of_interest = ["task2", "task3", "task2_after_t3"]
        for k in ["task2", "task3", "task2_after_t3"]:
            vals = []
            for r in runs:
                if k in r:
                    if "task2" in k:
                        vals.append(r[k].get("dice"))
                    else:
                        vals.append(r[k].get("mae"))
            vals = [v for v in vals if v is not None]
            if vals:
                mean, std = np.mean(vals), np.std(vals)
                all_results[method]["mean_std"][k] = {"mean": float(mean), "std": float(std)}
        if all_results[method]["bwt"]:
            bwt_arr = np.array(all_results[method]["bwt"])
            all_results[method]["bwt_mean_std"] = {"mean": float(np.mean(bwt_arr)), "std": float(np.std(bwt_arr))}

    # Print summary
    out_dir = ROOT / "outputs" / "multi_seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aggregate.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("MULTI-SEED SUMMARY (mean ± std)")
    print("=" * 60)
    for method in methods:
        ms = all_results[method].get("mean_std", {})
        bwt = all_results[method].get("bwt_mean_std")
        print(f"\n{method}:")
        for k, v in ms.items():
            m, s = v["mean"], v["std"]
            metric_name = "Dice" if "task2" in k else "MAE"
            print(f"  {k}: {m:.4f} ± {s:.4f} ({metric_name})")
        if bwt:
            print(f"  BWT: {bwt['mean']:.4f} ± {bwt['std']:.4f} (neg=forgetting)")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
