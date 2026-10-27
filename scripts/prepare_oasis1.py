#!/usr/bin/env python3
"""
Prepare OASIS-1 as three whole-head tasks (Task009_OASIS_AGE, Task010_OASIS_SEX,
Task011_OASIS_CDR).

Why. Brain age and sex are currently measured on IXI alone, so the claim that LoRA is the
only frozen-backbone strategy that works on a global task rests on one cohort and one
scanner set. OASIS-1 (416 subjects, 18-96 y, one 1.5T Siemens Vision) gives the same two
tasks on a second cohort, plus a clinical label (Clinical Dementia Rating) that IXI lacks.

Input is the challenge-supplied PROCESSED/MPRAGE/SUBJ_111 volume: the 3-4 MPRAGE repeats
averaged, N4 gain-field corrected, resampled to 1 mm isotropic in native space (no atlas
registration), so it is the closest counterpart to the 1 mm IXI pipeline. The reliability
re-scans (OAS1_xxxx_MR2) are dropped so every subject appears once. Each volume is
z-scored on its >0 mask, trimmed to content and resampled to one S^3 cube exactly as
scripts/prepare_ixi_wholebrain.py does, written once under Task009 and symlinked into
Task010/Task011.

Labels: age in years; sex 0 = male, 1 = female (the IXI convention); CDR task label
1 if CDR > 0 else 0, restricted to subjects aged >= 60 so that the demented and
non-demented classes overlap in age (younger subjects carry no CDR in OASIS-1 anyway).

  python scripts/prepare_oasis1.py --size 96
"""
import argparse
import glob
import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts/prepare_plan_b_data.py")
_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prep)
_wb_spec = importlib.util.spec_from_file_location("wb", ROOT / "scripts/prepare_ixi_wholebrain.py")
_wb = importlib.util.module_from_spec(_wb_spec)
_wb_spec.loader.exec_module(_wb)

from src.data import TASK_CONFIGS  # noqa: E402

CDR_MIN_AGE = 60


def read_demographics(txt: Path) -> dict:
    d = {}
    for line in txt.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        d[k.strip()] = v.strip()
    out = {"age": float(d["AGE"]) if d.get("AGE") else None,
           "sex": {"Male": 0, "Female": 1}.get(d.get("M/F")),
           "cdr": float(d["CDR"]) if d.get("CDR") not in (None, "") else None}
    return out


def find_sessions(root: Path):
    sessions = []
    for sdir in sorted(root.glob("disc*/OAS1_*_MR1")):
        sid = sdir.name[: -len("_MR1")]
        img = sorted(glob.glob(str(sdir / "PROCESSED/MPRAGE/SUBJ_111/*_sbj_111.img")))
        raw = sorted(glob.glob(str(sdir / "RAW/*_mpr-1_anon.img")))
        txt = sdir / f"{sdir.name}.txt"
        if not txt.exists() or not (img or raw):
            continue
        sessions.append({"id": sid, "img": Path(img[0]) if img else None,
                         "raw": Path(raw[0]) if raw else None, "txt": txt})
    return sessions


def load_1mm(sess) -> np.ndarray:
    """(1, D, H, W) float32 at 1 mm. SUBJ_111 is already 1 mm; RAW (1x1x1.25) is resampled."""
    if sess["img"] is not None:
        vol, _ = _prep._load_nifti(str(sess["img"]))
        vol = np.squeeze(np.asarray(vol, dtype=np.float32))
    else:
        vol, _ = _prep._load_nifti(str(sess["raw"]))
        vol = np.squeeze(np.asarray(vol, dtype=np.float32))
        sp = tuple(float(s) for s in _prep._get_spacing_from_nifti(str(sess["raw"]))[:3])
        if sp != _prep.TARGET_SPACING:
            vol = _prep._resample_volume(vol, sp, _prep.TARGET_SPACING)
    return _prep._zscore_normalize(vol[np.newaxis, ...])


def to_cube(vol: np.ndarray, S: int) -> np.ndarray:
    from scipy.ndimage import zoom
    vol = _wb.crop_to_content(vol)
    cube = zoom(vol, [1.0] + [S / s for s in vol.shape[1:]], order=1).astype(np.float32)
    pad = [(0, 0)] + [(0, max(0, S - d)) for d in cube.shape[1:]]
    return np.pad(cube, pad)[:, :S, :S, :S]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "data/plan_b_raw/oasis1"))
    ap.add_argument("--out_dir", default=str(ROOT / "data/preprocessed"))
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="first N subjects only (smoke test)")
    args = ap.parse_args()
    _prep._ensure_nibabel()

    age_dir = Path(args.out_dir) / TASK_CONFIGS[9]["name"]
    sex_dir = Path(args.out_dir) / TASK_CONFIGS[10]["name"]
    cdr_dir = Path(args.out_dir) / TASK_CONFIGS[11]["name"]
    for d in (age_dir, sex_dir, cdr_dir):
        d.mkdir(parents=True, exist_ok=True)

    sessions = find_sessions(Path(args.source))
    if not sessions:
        raise SystemExit(f"no OAS1_*_MR1 sessions with images under {args.source}")
    if args.limit:
        sessions = sessions[: args.limit]
    print(f"{len(sessions)} subjects -> {age_dir} at {args.size}^3", flush=True)

    n_age = n_sex = n_cdr = n_raw = 0
    pos = 0
    for i, s in enumerate(sessions, 1):
        demo = read_demographics(s["txt"])
        safe = re.sub(r"[^\w\-]", "_", s["id"])
        base = age_dir / f"sample_{safe}"
        npy = str(base) + ".npy"
        if not os.path.exists(npy):
            try:
                cube = to_cube(load_1mm(s), args.size)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {s['id']}: {type(exc).__name__}: {exc}", flush=True)
                continue
            np.save(npy, cube)
        n_raw += s["img"] is None
        # Task009: age. Every OASIS-1 subject has one.
        if demo["age"] is not None:
            np.savetxt(str(base) + ".txt", [demo["age"]], fmt="%.1f")
            n_age += 1
        # Task010 / Task011 share the cube by symlink; labels differ.
        for d, label, ok in ((sex_dir, demo["sex"], demo["sex"] is not None),
                             (cdr_dir, None if demo["cdr"] is None else int(demo["cdr"] > 0),
                              demo["cdr"] is not None and demo["age"] is not None and demo["age"] >= CDR_MIN_AGE)):
            if not ok:
                continue
            link = d / f"sample_{safe}.npy"
            if not link.exists():
                os.symlink(os.path.relpath(npy, d), link)
            np.savetxt(str(d / f"sample_{safe}.txt"), [label], fmt="%d")
            if d is sex_dir:
                n_sex += 1
            else:
                n_cdr += 1
                pos += label
        if i % 50 == 0 or i == len(sessions):
            print(f"  {i}/{len(sessions)}", flush=True)

    for d in (age_dir, sex_dir, cdr_dir):
        c = d / ".usable_cases.json"
        if c.exists():
            c.unlink()
    print(f"done: age {n_age}, sex {n_sex}, cdr {n_cdr} (CDR>0: {pos}); {n_raw} fell back to RAW mpr-1")


if __name__ == "__main__":
    main()
