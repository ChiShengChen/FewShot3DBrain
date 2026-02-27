#!/usr/bin/env python3
"""
Aggregate MICCAI experiment results into JSON for the paper.
Usage: python scripts/aggregate_miccai_results.py [n32|n64]
Output: outputs/miccai_experiments/<phase>/aggregate.json
"""
import argparse
import json
import statistics
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
MULTI_SEED = ROOT / "outputs" / "multi_seed"
SEEDS = [42, 43, 44]


def load_metrics(base: Path, method: str, seed: int) -> Optional[dict]:
    """Load metrics.json for a method+seed."""
    if method in ("sequential_linear", "sequential_ft"):
        mp = base / method / f"seed{seed}" / method / "metrics.json"
    else:
        mp = base / method / f"seed{seed}" / "metrics.json"
    if not mp.exists():
        return None
    with open(mp) as f:
        return json.load(f)


def aggregate_method(base: Path, method: str, phase: str = "n32") -> Optional[dict]:
    """Aggregate runs for one method across seeds.
    phase: n32/n64 use task2_after_t3 (BWT for T2); t32 uses task3_after_t2 (BWT for T3)."""
    runs = []
    for seed in SEEDS:
        m = load_metrics(base, method, seed)
        if m is None:
            return None
        runs.append(m)

    def mean_std(key: str, subkey: str) -> Tuple[float, float]:
        vals = [r[key][subkey] for r in runs if key in r and subkey in r[key]]
        if not vals:
            return 0.0, 0.0
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

    out = {"runs": runs}
    out["mean_std"] = {
        "task2": {"mean": mean_std("task2", "dice")[0], "std": mean_std("task2", "dice")[1]},
        "task3": {"mean": mean_std("task3", "mae")[0], "std": mean_std("task3", "mae")[1]},
    }
    if phase in ("t32", "t32_t3t2") and "task3_after_t2" in runs[0]:
        m, s = mean_std("task3_after_t2", "mae")
        out["mean_std"]["task3_after_t2"] = {"mean": m, "std": s}
        out["bwt"] = [r["task3_after_t2"]["mae"] - r["task3"]["mae"] for r in runs]
        out["bwt_mean_std"] = {
            "mean": statistics.mean(out["bwt"]),
            "std": statistics.stdev(out["bwt"]) if len(out["bwt"]) > 1 else 0.0,
        }
    elif "task2_after_t3" in runs[0]:
        m, s = mean_std("task2_after_t3", "dice")
        out["mean_std"]["task2_after_t3"] = {"mean": m, "std": s}
        out["bwt"] = [r["task2_after_t3"]["dice"] - r["task2"]["dice"] for r in runs]
        out["bwt_mean_std"] = {
            "mean": statistics.mean(out["bwt"]),
            "std": statistics.stdev(out["bwt"]) if len(out["bwt"]) > 1 else 0.0,
        }
    else:
        out["bwt"] = [0.0] * len(runs)
        out["bwt_mean_std"] = {"mean": 0.0, "std": 0.0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", nargs="?", default="n32", choices=["n32", "n64", "t32", "t32_t3t2"],
                    help="n32, n64, t32 (old), or t32_t3t2 (T3→T2, default: n32)")
    args = ap.parse_args()

    base = ROOT / "outputs" / "miccai_experiments" / args.phase
    methods = ["lora", "sequential_linear", "sequential_ft", "ewc", "lwf", "replay"]
    if args.phase == "n64":
        methods = ["lora", "sequential_linear", "sequential_ft", "ewc", "lwf"]  # no replay in phase 2
    elif args.phase in ("t32", "t32_t3t2"):
        methods = ["lora", "sequential_linear", "sequential_ft"]  # phase 3 only
    result = {}

    for method in methods:
        agg = aggregate_method(base, method, phase=args.phase)
        if agg is None and args.phase == "n32" and method in ("sequential_linear", "sequential_ft"):
            agg = aggregate_method(MULTI_SEED, method, phase=args.phase)
        if agg is not None:
            result[method] = agg

    out_path = base / "aggregate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
