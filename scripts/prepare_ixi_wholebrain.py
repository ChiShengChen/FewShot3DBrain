#!/usr/bin/env python3
"""
Prepare IXI brain age as a *whole-brain* regression task (Task005_IXI_WB).

Under the patch protocol no method beat a constant predictor on brain age: a 64^3
window of a 240x240x180 head sees a few percent of the brain, and global age cues
(ventricle size, cortical atrophy) are not local. This writes each subject's full
two-channel volume resampled to a fixed cube so one forward pass sees the whole head,
reusing the existing regression head and evaluation path unchanged (the "patch" is the
whole volume).

Reads the already-preprocessed Task003_FOMO3 volumes (1 mm, z-scored) and their repaired
age labels; writes sample_<id>.npy of shape (2, S, S, S) and copies the .txt label.

  python scripts/prepare_ixi_wholebrain.py --size 96
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import TASK_CONFIGS, list_case_files  # noqa: E402


def crop_to_content(vol: np.ndarray, margin: int = 4) -> np.ndarray:
    """Trim empty border so the resampled cube spends its voxels on the head, not air."""
    mask = np.abs(vol).sum(axis=0) > 1e-3
    if not mask.any():
        return vol
    idx = np.where(mask)
    lo = [max(0, int(i.min()) - margin) for i in idx]
    hi = [min(s, int(i.max()) + 1 + margin) for i, s in zip(idx, vol.shape[1:])]
    return vol[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "data/preprocessed"))
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from scipy.ndimage import zoom

    src_files = list_case_files(args.data_dir, 3)          # usable ages only
    if not src_files:
        raise SystemExit("no usable Task003 cases")
    if args.limit:
        src_files = src_files[: args.limit]

    out_dir = Path(args.data_dir) / TASK_CONFIGS[5]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    S = args.size
    print(f"{len(src_files)} subjects -> {out_dir} at {S}^3", flush=True)

    for i, f in enumerate(src_files, 1):
        base = os.path.basename(f)
        dst = out_dir / base
        if (dst.with_suffix(".npy")).exists() and (dst.with_suffix(".txt")).exists():
            continue
        vol = np.load(f + ".npy").astype(np.float32)          # (2, D, H, W)
        vol = crop_to_content(vol)
        factors = [1.0] + [S / s for s in vol.shape[1:]]
        cube = zoom(vol, factors, order=1).astype(np.float32)
        # zoom can be off by one voxel from rounding; pad/crop to exactly S^3
        pad = [(0, 0)] + [(0, max(0, S - d)) for d in cube.shape[1:]]
        cube = np.pad(cube, pad)[:, :S, :S, :S]
        np.save(str(dst) + ".npy", cube)
        shutil.copyfile(f + ".txt", str(dst) + ".txt")
        if i % 50 == 0 or i == len(src_files):
            print(f"  {i}/{len(src_files)}  {tuple(cube.shape)}", flush=True)

    cache = out_dir / ".usable_cases.json"
    if cache.exists():
        cache.unlink()
    print("done")


if __name__ == "__main__":
    main()
