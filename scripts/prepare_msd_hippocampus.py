#!/usr/bin/env python3
"""
Prepare MSD Task04 Hippocampus as a third segmentation task (Task007_HIPPO).

Why this task. The finding that full-decoder linear probing edges out LoRA on segmentation
rests on two tumour/lesion datasets that share a pipeline and a modality set. Hippocampus
segmentation differs on every axis that could matter: a single T1 channel rather than three,
a healthy anatomical structure rather than a pathology, a large and consistent foreground
rather than a small variable lesion, and volumes (~35x51x35) smaller than the 64^3 window,
so the whole structure is visible in one forward pass. If the segmentation conclusion holds
here too, it is about the presence of a large task-specific decoder rather than about
tumours.

Labels are merged to binary (anterior + posterior head -> foreground), matching the binary
treatment of BraTS and ISLES.

  python scripts/prepare_msd_hippocampus.py
"""
import argparse
import json
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
    ap.add_argument("--source", default=str(ROOT / "data/plan_b_raw/msd/Task04_Hippocampus"))
    ap.add_argument("--out_dir", default=str(ROOT / "data/preprocessed"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import nibabel as nib

    src = Path(args.source)
    task_dir = Path(args.out_dir) / TASK_CONFIGS[7]["name"]
    task_dir.mkdir(parents=True, exist_ok=True)

    # macOS archive artefacts ("._name") are not volumes; skip them.
    images = sorted(f for f in (src / "imagesTr").glob("*.nii.gz") if not f.name.startswith("._"))
    if not images:
        raise SystemExit(f"no images under {src / 'imagesTr'}")
    if args.limit:
        images = images[: args.limit]
    print(f"{len(images)} cases -> {task_dir}", flush=True)

    n_ok = n_empty = 0
    for i, img in enumerate(images, 1):
        lab = src / "labelsTr" / img.name
        if not lab.exists():
            continue
        try:
            vol = np.asarray(nib.load(str(img)).dataobj, dtype=np.float32)
            seg = np.asarray(nib.load(str(lab)).dataobj)
            seg = (seg > 0).astype(np.float32)                  # anterior + posterior
            if not seg.any():
                n_empty += 1
                continue
            # already 1 mm isotropic; z-score over non-zero voxels, as for the other tasks
            mask = vol > 0
            if mask.sum():
                vol = np.where(mask, (vol - vol[mask].mean()) / (vol[mask].std() + 1e-8), vol)
            vol = vol[np.newaxis, ...].astype(np.float32)        # (1, D, H, W)
            seg = seg[np.newaxis, ...]

            safe = re.sub(r"[^\w-]", "_", img.name.split(".")[0])
            base = task_dir / f"sample_{safe}"
            np.save(str(base) + ".npy", vol)
            np.save(str(base) + "_seg.npy", seg)
            n_ok += 1
            if i % 50 == 0 or i == len(images):
                print(f"  {i}/{len(images)}  {tuple(vol.shape)}  fg {int(seg.sum())}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {img.name}: {type(exc).__name__}: {exc}", flush=True)

    cache = task_dir / ".usable_cases.json"
    if cache.exists():
        cache.unlink()
    print(f"done: {n_ok} written, {n_empty} skipped for empty mask")


if __name__ == "__main__":
    main()
