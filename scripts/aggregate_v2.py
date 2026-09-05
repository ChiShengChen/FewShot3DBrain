#!/usr/bin/env python3
"""
Aggregate corrected-protocol results into the tables the paper needs.

  python scripts/aggregate_v2.py                 # markdown summary
  python scripts/aggregate_v2.py --latex         # IEEE-style booktabs table
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

PRETTY = {
    "lora": "LoRA (enc+dec)",
    "lora_encoder_only": "LoRA (enc only)",
    "sequential_linear": "Sequential Linear",
    "sequential_ft": "Sequential FT",
    "pertask_lora": "Per-task LoRA (oracle)",
}


def collect(root, n_shot):
    runs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(root, f"n{n_shot}", "*", "seed*", "results.json"))):
        with open(f) as fh:
            r = json.load(fh)
        runs[r["method"]].append(r)
    return runs


def ms(vals):
    """mean +/- std over seeds, ignoring configurations that never produced the metric."""
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    return float(v.mean()), float(v.std()), int(v.size)


def fmt(stat, digits=3):
    if stat is None:
        return "--"
    m, s, n = stat
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def summarise(runs):
    rows = []
    for method, rs in runs.items():
        seg = [r["per_task"].get("task2", {}) for r in rs]
        reg = [r["per_task"].get("task3", {}) for r in rs]

        dice = ms([s.get("test", {}).get("dice") for s in seg])
        hd = ms([s.get("test", {}).get("hd95") for s in seg])
        mae = ms([s.get("test", {}).get("mae") for s in reg])
        r2 = ms([s.get("test", {}).get("r2") for s in reg])
        slope = ms([s.get("test", {}).get("calibration_slope") for s in reg])

        after = [s.get("test_after_task3", {}).get("dice") for s in seg]
        after_stat = ms(after)
        if after_stat is None:                       # adapter isolation: nothing to re-score
            bwt_stat, after_txt = (0.0, 0.0, len(rs)), "= T1 (isolated)"
        else:
            bwt_stat = ms([a - s.get("test", {}).get("dice", np.nan)
                           for a, s in zip(after, seg) if a is not None])
            after_txt = fmt(after_stat)

        params = [s.get("params", {}).get("trainable_pct") for s in seg if s.get("params")]
        rows.append({
            "method": PRETTY.get(method, method),
            "key": method,
            "n_seeds": len(rs),
            "dice": dice, "hd95": hd, "mae": mae, "r2": r2, "slope": slope,
            "after": after_txt, "bwt": bwt_stat,
            "params_pct": float(np.mean(params)) if params else None,
        })
    order = ["lora", "lora_encoder_only", "sequential_linear", "sequential_ft", "pertask_lora"]
    rows.sort(key=lambda r: order.index(r["key"]) if r["key"] in order else 99)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "outputs/benchmark_v2"))
    ap.add_argument("--n_shot", type=int, default=32)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    runs = collect(args.root, args.n_shot)
    if not runs:
        raise SystemExit(f"No results under {args.root}/n{args.n_shot}. Run scripts/run_stage.sh first.")
    rows = summarise(runs)

    if args.latex:
        print(r"\begin{tabular}{lcccccc}")
        print(r"\toprule")
        print(r"Method & Dice$\uparrow$ & HD95$\downarrow$ & MAE (yr)$\downarrow$ & $R^2\uparrow$ "
              r"& Dice after T2 & BWT \\")
        print(r"\midrule")
        for r in rows:
            print(f"{r['method']} & {fmt(r['dice'])} & {fmt(r['hd95'], 1)} & {fmt(r['mae'], 2)} "
                  f"& {fmt(r['r2'])} & {r['after']} & {fmt(r['bwt'])} \\\\".replace("±", r"$\pm$"))
        print(r"\bottomrule")
        print(r"\end{tabular}")
        return

    print(f"\n=== Corrected protocol, n_shot={args.n_shot} "
          f"(held-out test set, whole-volume inference) ===\n")
    hdr = ["Method", "seeds", "T1 Dice", "T1 HD95", "T2 MAE(yr)", "T2 R2", "T2 slope",
           "T1 after T2", "BWT", "train%"]
    print(" | ".join(hdr))
    print("-|-".join("-" * len(h) for h in hdr))
    for r in rows:
        print(" | ".join([
            r["method"], str(r["n_seeds"]), fmt(r["dice"]), fmt(r["hd95"], 1),
            fmt(r["mae"], 2), fmt(r["r2"]), fmt(r["slope"], 2),
            r["after"], fmt(r["bwt"]),
            f"{r['params_pct']:.3f}" if r["params_pct"] is not None else "--",
        ]))
    print("\nNotes:")
    print("  * MAE is in years against real IXI ages (placeholder-age subjects excluded).")
    print("  * R2/slope near 0 means the model is predicting close to a constant.")
    print("  * Dice/HD95 are per-case means from sliding-window inference over full volumes.")


if __name__ == "__main__":
    main()
