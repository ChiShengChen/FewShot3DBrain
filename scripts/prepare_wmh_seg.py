#!/usr/bin/env python3
"""
Prepare the WMH Segmentation Challenge 2017 as Task008_WMH_SEG.

Why. BraTS, ISLES and MSD hippocampus are each one cohort and (effectively) one
acquisition. WMH 2017 has 170 subjects from five scanners at three sites (Utrecht,
Singapore, Amsterdam), so it adds a pathology the paper does not cover and a real
multi-site shift inside one task.

Channels: FLAIR and T1 from the challenge's pre/ folder (bias-field corrected, T1 already
registered to the FLAIR grid). Label: wmh.nii.gz, where 1 = WMH and 2 = "other pathology";
the challenge scores WMH only, so 2 is mapped to background. Volumes are resampled to the
project's 1 mm grid and z-scored per channel on the >0 mask, matching the other tasks.
Output: sample_<site>_<id>.npy (2, D, H, W) and sample_<site>_<id>_seg.npy (1, D, H, W).

  python scripts/prepare_wmh_seg.py --source data/plan_b_raw/wmh
"""
import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts/prepare_plan_b_data.py")
_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep)

from src.data import TASK_CONFIGS  # noqa: E402


def find_cases(root: Path):
    cases = []
    for wmh in sorted(root.rglob("wmh.nii.gz")):
        sub = wmh.parent
        flair, t1 = sub / "pre" / "FLAIR.nii.gz", sub / "pre" / "T1.nii.gz"
        if not (flair.exists() and t1.exists()):
            continue
        # <split>/<site>[/<scanner>]/<id>/wmh.nii.gz -> site + id
        rel = sub.relative_to(root).parts
        site = rel[1] if len(rel) >= 3 else rel[0]
        cases.append({"id": f"{site}_{sub.name}", "split": rel[0], "scanner": rel[2] if len(rel) == 4 else site,
                      "flair": flair, "t1": t1, "wmh": wmh})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "data/plan_b_raw/wmh"))
    ap.add_argument("--out_dir", default=str(ROOT / "data/preprocessed"))
    ap.add_argument("--limit", type=int, default=0, help="prepare only the first N cases (smoke test)")
    args = ap.parse_args()

    _prep._ensure_nibabel()
    task_dir = Path(args.out_dir) / TASK_CONFIGS[8]["name"]
    task_dir.mkdir(parents=True, exist_ok=True)

    cases = find_cases(Path(args.source))
    if not cases:
        raise SystemExit(f"no WMH cases with pre/FLAIR + pre/T1 + wmh under {args.source}")
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} cases -> {task_dir}", flush=True)

    n_ok = n_empty = 0
    manifest = []
    for i, c in enumerate(cases, 1):
        try:
            flair, _ = _prep._load_nifti(str(c["flair"]))
            t1 = _prep._load_nifti(str(c["t1"]))[0]
            lab = _prep._load_nifti(str(c["wmh"]))[0]
            flair, t1, lab = (np.squeeze(np.asarray(v, dtype=np.float32)) for v in (flair, t1, lab))
            ref_shape = tuple(flair.shape)[:3]
            sp = tuple(float(s) for s in _prep._get_spacing_from_nifti(str(c["flair"]))[:3])
            t1 = _prep._align_vol_to_ref(t1, ref_shape)
            lab = _prep._align_vol_to_ref(lab, ref_shape)
            msk = (np.abs(lab - 1.0) < 0.5).astype(np.float32)      # 1 = WMH; 2 = other pathology -> 0

            vol = np.stack([flair, t1], axis=0).astype(np.float32)
            if sp != _prep.TARGET_SPACING:
                vol4d = np.moveaxis(vol, 0, -1)
                vol4d = _prep._resample_volume(vol4d, sp + (1,), _prep.TARGET_SPACING + (1,))
                vol = np.moveaxis(vol4d, -1, 0)
                msk = _prep._resample_volume(msk, sp, _prep.TARGET_SPACING)
            seg = (msk > 0.5).astype(np.float32)[np.newaxis, ...]
            if seg.shape[1:] != vol.shape[1:]:
                seg = (_prep._align_vol_to_ref(seg[0], vol.shape[1:]) > 0.5).astype(np.float32)[np.newaxis, ...]
            if not np.any(seg > 0):
                n_empty += 1
                continue

            vol = _prep._zscore_normalize(vol)
            safe = re.sub(r"[^\w\-]", "_", c["id"])[:64]
            base = task_dir / f"sample_{safe}"
            np.save(str(base) + ".npy", vol.astype(np.float32))
            np.save(str(base) + "_seg.npy", seg)
            manifest.append(f"{safe}\t{c['split']}\t{c['scanner']}\t{int(seg.sum())}")
            n_ok += 1
            if i % 25 == 0 or i == len(cases):
                print(f"  {i}/{len(cases)}  shape {tuple(vol.shape)}  wmh voxels {int(seg.sum())}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {c['id']}: {type(exc).__name__}: {exc}", flush=True)

    (task_dir / "manifest.tsv").write_text("id\tchallenge_split\tscanner\twmh_voxels_1mm\n" + "\n".join(manifest) + "\n")
    print(f"done: {n_ok} written, {n_empty} skipped for empty mask")
    cache = task_dir / ".usable_cases.json"
    if cache.exists():
        cache.unlink()


if __name__ == "__main__":
    main()
