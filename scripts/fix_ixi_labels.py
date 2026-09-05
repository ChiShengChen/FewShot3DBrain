#!/usr/bin/env python3
"""
Repair Task003 (IXI brain age) labels from IXI.xls.

The original preprocessing matched subject IDs as strings, so zero-padded IXI IDs
below 100 (e.g. filename "IXI002" -> "002" vs. spreadsheet IXI_ID 2) never matched
and silently fell back to the placeholder age of 50.0. This rewrites every
sample_IXI-<id>.txt with the real AGE, and writes "nan" for subjects that have no
usable age so that the loader can exclude them.

Usage:
  python scripts/fix_ixi_labels.py --dry-run
  python scripts/fix_ixi_labels.py
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_ages(xls_path: Path) -> dict:
    """Map integer IXI_ID -> age in years (None when the spreadsheet has no age)."""
    import pandas as pd

    engine = "xlrd" if xls_path.suffix.lower() == ".xls" else None
    df = pd.read_excel(xls_path, engine=engine)
    if "IXI_ID" not in df.columns or "AGE" not in df.columns:
        raise SystemExit(f"Unexpected columns in {xls_path}: {list(df.columns)}")
    ages = {}
    for row in df.itertuples():
        age = float(row.AGE) if pd.notna(row.AGE) else None
        ages[int(row.IXI_ID)] = age
    return ages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_dir", default=str(ROOT / "data/preprocessed/Task003_FOMO3"))
    ap.add_argument("--xls", default=str(ROOT / "data/plan_b_raw/ixi/IXI.xls"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    task_dir = Path(args.task_dir)
    xls = Path(args.xls)
    if not task_dir.is_dir():
        raise SystemExit(f"No such task dir: {task_dir}")
    if not xls.exists():
        raise SystemExit(f"No IXI.xls at {xls}")

    ages = load_ages(xls)
    files = sorted(glob.glob(str(task_dir / "sample_IXI-*.txt")))
    if not files:
        raise SystemExit(f"No label files under {task_dir}")

    report = {"repaired": [], "unchanged": 0, "excluded_no_row": [], "excluded_nan_age": []}
    for f in files:
        m = re.search(r"IXI-(\d+)", os.path.basename(f))
        if not m:
            continue
        sid = int(m.group(1))
        try:
            current = float(np.loadtxt(f))
        except Exception:
            current = float("nan")

        if sid not in ages:
            report["excluded_no_row"].append(sid)
            new = float("nan")
        elif ages[sid] is None:
            report["excluded_nan_age"].append(sid)
            new = float("nan")
        else:
            new = ages[sid]

        same = (np.isnan(current) and np.isnan(new)) or (
            np.isfinite(current) and np.isfinite(new) and abs(current - new) < 1e-4
        )
        if same:
            report["unchanged"] += 1
            continue
        report["repaired"].append({"id": sid, "old": current, "new": new})
        if not args.dry_run:
            np.savetxt(f, [new], fmt="%.4f")

    usable = [a for a in (ages.get(int(re.search(r"IXI-(\d+)", os.path.basename(f)).group(1)))
                          for f in files) if a is not None]
    arr = np.array(usable, dtype=float)

    print(f"label files          : {len(files)}")
    print(f"rewritten            : {len(report['repaired'])}")
    print(f"already correct      : {report['unchanged']}")
    print(f"excluded (no xls row): {len(report['excluded_no_row'])} -> {report['excluded_no_row']}")
    print(f"excluded (NaN age)   : {len(report['excluded_nan_age'])} -> {report['excluded_nan_age']}")
    print(f"usable subjects      : {len(arr)}  age mean={arr.mean():.2f} std={arr.std():.2f} "
          f"range={arr.min():.1f}-{arr.max():.1f}")

    if not args.dry_run:
        out = task_dir.parent / "ixi_label_repair_report.json"
        out.write_text(json.dumps(report, indent=2))
        print(f"\nreport -> {out}")
    else:
        print("\n(dry run; nothing written)")


if __name__ == "__main__":
    sys.exit(main())
