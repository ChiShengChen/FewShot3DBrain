#!/usr/bin/env python3
"""
Run ablations: shot count (16/32/64), LoRA rank (4/8/16).
Records: BWT, Avg accuracy, GPU memory, training time, params.
"""
import sys
import json
import subprocess
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))

METHODS = {
    "lora": ("run_continual_lora.py", ["--lora_decoder"]),
    "sequential_linear": ("run_continual_baselines.py", ["--baseline", "sequential_linear"]),
    "sequential_ft": ("run_continual_baselines.py", ["--baseline", "sequential_ft"]),
}
SHOTS = [16, 32, 64]
LORA_RANKS = [4, 8, 16]
MAE_SCALE = 20.0


def run_single(args, method, seed, n_shot, lora_r=None):
    script, extra = METHODS[method]
    subdir = f"seed{seed}_shot{n_shot}" + (f"_r{lora_r}" if method == "lora" and lora_r and lora_r != 8 else "")
    base_dir = str(ROOT / "outputs" / "ablations" / method / subdir)
    cmd = [sys.executable, str(ROOT / script), "--tasks", "2", "3",
           "--data_dir", args.data_dir, "--save_dir", base_dir,
           "--n_shot", str(n_shot), "--epochs", str(args.epochs), "--seed", str(seed)]
    if args.pretrained_path:
        cmd.extend(["--pretrained_path", args.pretrained_path])
    if method == "lora" and lora_r is not None:
        cmd.extend(["--lora_r", str(lora_r)])
    cmd.extend(extra)
    print(f"\nRunning {method} seed={seed} n_shot={n_shot}" + (f" r={lora_r}" if lora_r else ""))
    ret = subprocess.run(cmd, cwd=str(ROOT))
    if ret.returncode != 0:
        return {}
    if method == "lora":
        path = Path(base_dir) / "metrics.json"
    else:
        bl = "sequential_linear" if "linear" in method else "sequential_ft"
        path = Path(base_dir) / bl / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def compute_bwt(m):
    r22 = m.get("task2", {}).get("dice")
    r32 = m.get("task2_after_t3", {}).get("dice") or r22
    return (r32 - r22) if r22 is not None and r32 is not None else None


def compute_avg_acc(m):
    d2, m3 = m.get("task2", {}).get("dice"), m.get("task3", {}).get("mae")
    vals = ([d2] if d2 is not None else []) + ([1 - min(m3 / MAE_SCALE, 1)] if m3 is not None else [])
    return float(np.mean(vals)) if vals else None


def aggregate_resources(m):
    t, g, p = [], [], []
    for k in ["task2", "task3"]:
        if k in m:
            if m[k].get("time_sec") is not None:
                t.append(m[k]["time_sec"])
            if m[k].get("gpu_max_mb") is not None:
                g.append(m[k]["gpu_max_mb"])
            if m[k].get("n_params") is not None:
                p.append(m[k]["n_params"])
    return {"time": sum(t) if t else None, "gpu": max(g) if g else None, "params": sum(p) if p else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations", nargs="+", default=["shot", "lora_rank"], choices=["shot", "lora_rank"])
    ap.add_argument("--method", default="all", choices=["all", "lora", "sequential_linear", "sequential_ft"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--pretrained_path", default="./weights/fomo25_mmunetvae_pretrained.ckpt")
    ap.add_argument("--data_dir", default="./data/preprocessed")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    methods = ["lora", "sequential_linear", "sequential_ft"] if args.method == "all" else [args.method]
    shots = SHOTS if "shot" in args.ablations else [32]
    ranks = LORA_RANKS if "lora_rank" in args.ablations else [8]

    out_dir = ROOT / "outputs" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for method in methods:
        results[method] = {}
        for n_shot in shots:
            rlist = ranks if method == "lora" else [None]
            for r in rlist:
                key = f"shot{n_shot}" + (f"_r{r}" if r is not None else "")
                data = {"bwt": [], "avg_acc": [], "time": [], "gpu": [], "params": []}
                for seed in args.seeds:
                    m = run_single(args, method, seed, n_shot, lora_r=r)
                    if not m:
                        continue
                    b = compute_bwt(m)
                    if b is not None:
                        data["bwt"].append(b)
                    a = compute_avg_acc(m)
                    if a is not None:
                        data["avg_acc"].append(a)
                    res = aggregate_resources(m)
                    if res["time"]:
                        data["time"].append(res["time"])
                    if res["gpu"]:
                        data["gpu"].append(res["gpu"])
                    if res["params"]:
                        data["params"].append(res["params"])
                results[method][key] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in data.items() if v}

    out_path = out_dir / "ablation_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nAblation summary saved to", out_path)
    for method in methods:
        for key, d in results[method].items():
            s = ", ".join(f"{k}={d[k]['mean']:.4f}±{d[k]['std']:.4f}" for k in d)
            print(f"  {method} {key}: {s}")


if __name__ == "__main__":
    main()
