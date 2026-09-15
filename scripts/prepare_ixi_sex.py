#!/usr/bin/env python3
"""
Prepare IXI sex classification on whole-head input (Task006_IXI_SEX).

Why this task. The paper's central claim is that adapters are *necessary*, not merely
economical, when a task has no large head to train -- evidenced so far only by brain-age
regression. One task cannot carry that claim. Sex prediction is a second global task with
the same structure (pooled encoder features, a tiny head) but a different output type
(binary classification rather than regression), on the same subjects, scans and modalities
as brain age. If linear probing and BitFit collapse here too while LoRA does not, the
finding is about capacity placement; if they do not, it was about regression.

It also restores the classification task type that was dropped when ISLES-vs-IXI infarct
detection proved confounded by acquisition: here both classes come from the same cohort
and the same scans, so only the label differs.

Volumes are the 96^3 whole-head cubes already built for Task005; they are symlinked rather
than copied. Labels come from IXI.xls SEX_ID (1=m -> 0, 2=f -> 1).

  python scripts/prepare_ixi_sex.py
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import TASK_CONFIGS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "data/preprocessed"))
    ap.add_argument("--xls", default=str(ROOT / "data/plan_b_raw/ixi/IXI.xls"))
    args = ap.parse_args()

    import pandas as pd

    src = Path(args.data_dir) / TASK_CONFIGS[5]["name"]          # whole-head cubes
    dst = Path(args.data_dir) / TASK_CONFIGS[6]["name"]
    if not src.is_dir():
        raise SystemExit(f"{src} missing; run scripts/prepare_ixi_wholebrain.py first")
    dst.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.xls, engine="xlrd")
    sex = {}
    for row in df.itertuples():
        try:
            v = int(row._2) if hasattr(row, "_2") else int(getattr(row, "SEX_ID (1=m, 2=f)"))
        except (TypeError, ValueError):
            continue
        if v in (1, 2):
            sex[int(row.IXI_ID)] = v - 1                          # 1=m -> 0, 2=f -> 1

    n_ok = n_missing = 0
    counts = {0: 0, 1: 0}
    for npy in sorted(src.glob("sample_IXI-*.npy")):
        sid = int(re.search(r"IXI-(\d+)", npy.name).group(1))
        if sid not in sex:
            n_missing += 1
            continue
        base = dst / npy.stem
        link = str(base) + ".npy"
        if not os.path.exists(link):
            os.symlink(os.path.relpath(npy, dst), link)           # 3.7 GB not duplicated
        np.savetxt(str(base) + ".txt", [sex[sid]], fmt="%d")
        counts[sex[sid]] += 1
        n_ok += 1

    cache = dst / ".usable_cases.json"
    if cache.exists():
        cache.unlink()
    print(f"{n_ok} subjects written ({n_missing} without a sex label)")
    print(f"  class balance: male {counts[0]}, female {counts[1]} "
          f"(majority-class accuracy {max(counts.values()) / max(n_ok, 1):.3f})")


if __name__ == "__main__":
    main()
