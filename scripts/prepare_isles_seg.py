#!/usr/bin/env python3
"""
Prepare ISLES 2022 as a lesion *segmentation* task (Task004_ISLES_SEG).

Task 1 reused ISLES for infarct classification against IXI healthy controls, which is
confounded: the two classes come from different acquisitions. Segmentation within the
ISLES cohort alone has no such problem - every case has an expert lesion mask - and it
gives the continual sequence a third task whose output type (a mask) matches Task 2, so
the segmentation head configuration can be tested twice.

Channels: DWI, ADC, FLAIR (the FLAIR is resampled onto the DWI grid). Output matches the
other tasks: sample_<id>.npy (3, D, H, W) float32 z-scored, sample_<id>_seg.npy (1, D, H, W).

  python scripts/prepare_isles_seg.py --source data/plan_b_raw/isles/ISLES-2022
"""
import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the resampling / normalisation helpers rather than re-implementing them.
_spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts/prepare_plan_b_data.py")
_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep)

from src.data import TASK_CONFIGS  # noqa: E402


def find_cases(root: Path):
    cases = []
    for sub in sorted(root.glob("sub-strokecase*")):
        if not sub.is_dir():
            continue
        dwi = sorted(sub.rglob("*_dwi.nii*"))
        adc = sorted(sub.rglob("*_adc.nii*"))
        flair = sorted(sub.rglob("*_FLAIR.nii*")) + sorted(sub.rglob("*_flair.nii*"))
        msk = sorted((root / "derivatives" / sub.name).rglob("*_msk.nii*"))
        if dwi and adc and flair and msk:
            cases.append({"id": sub.name, "dwi": dwi[0], "adc": adc[0], "flair": flair[0], "msk": msk[0]})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "data/plan_b_raw/isles/ISLES-2022"))
    ap.add_argument("--out_dir", default=str(ROOT / "data/preprocessed"))
    ap.add_argument("--limit", type=int, default=0, help="prepare only the first N cases (smoke test)")
    args = ap.parse_args()

    _prep._ensure_nibabel()
    src = Path(args.source)
    task_dir = Path(args.out_dir) / TASK_CONFIGS[4]["name"]
    task_dir.mkdir(parents=True, exist_ok=True)

    cases = find_cases(src)
    if not cases:
        raise SystemExit(f"no ISLES cases with DWI+ADC+FLAIR+mask under {src}")
    if args.limit:
        cases = cases[: args.limit]
    print(f"{len(cases)} cases -> {task_dir}", flush=True)

    n_ok = n_empty = 0
    for i, c in enumerate(cases, 1):
        try:
            dwi, _ = _prep._load_nifti(str(c["dwi"]))
            adc = _prep._load_nifti(str(c["adc"]))[0]
            flair = _prep._load_nifti(str(c["flair"]))[0]
            msk = _prep._load_nifti(str(c["msk"]))[0]
            ref_shape = tuple(dwi.shape)[:3]
            ref_spacing = _prep._get_spacing_from_nifti(str(c["dwi"]))

            # ADC and the mask share the DWI grid; FLAIR is a separate acquisition and
            # is brought onto it. The mask is resampled with nearest-neighbour semantics
            # via the >0.5 threshold after linear interpolation.
            adc = _prep._align_vol_to_ref(adc, ref_shape)
            flair = _prep._align_vol_to_ref(flair, ref_shape)
            msk = _prep._align_vol_to_ref(msk, ref_shape)

            vol = np.stack([dwi, adc, flair], axis=0).astype(np.float32)
            if _prep.TARGET_SPACING != ref_spacing:
                vol4d = np.moveaxis(vol, 0, -1)
                vol4d = _prep._resample_volume(vol4d, ref_spacing + (1,), _prep.TARGET_SPACING + (1,))
                vol = np.moveaxis(vol4d, -1, 0)
                msk = _prep._resample_volume(msk.astype(np.float32), ref_spacing, _prep.TARGET_SPACING)
            seg = (msk > 0.5).astype(np.float32)[np.newaxis, ...]
            if seg.shape[1:] != vol.shape[1:]:
                seg = _prep._align_vol_to_ref(seg[0], vol.shape[1:])[np.newaxis, ...]
                seg = (seg > 0.5).astype(np.float32)
            if not np.any(seg > 0):
                n_empty += 1
                continue

            vol = _prep._zscore_normalize(vol)
            safe = re.sub(r"[^\w\-]", "_", c["id"])[:64]
            base = task_dir / f"sample_{safe}"
            np.save(str(base) + ".npy", vol.astype(np.float32))
            np.save(str(base) + "_seg.npy", seg.astype(np.float32))
            n_ok += 1
            if i % 25 == 0 or i == len(cases):
                print(f"  {i}/{len(cases)}  shape {tuple(vol.shape)}  lesion voxels {int(seg.sum())}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {c['id']}: {type(exc).__name__}: {exc}", flush=True)

    print(f"done: {n_ok} written, {n_empty} skipped for empty mask")
    # Invalidate the usable-case cache so the new files are picked up.
    cache = task_dir / ".usable_cases.json"
    if cache.exists():
        cache.unlink()


if __name__ == "__main__":
    main()
