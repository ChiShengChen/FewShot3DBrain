#!/usr/bin/env python3
"""
Prepare Plan B datasets for FewShot3DBrain (MICCAI 2026).

Converts public datasets (ISLES 2022, BraTS Meningioma/Glioma, IXI) into the
format expected by FewShotFOMODataset.

Usage:
  python scripts/prepare_plan_b_data.py --task all --download
  python scripts/prepare_plan_b_data.py --task 1 --source_dir /path/to/ISLES-2022
  python scripts/prepare_plan_b_data.py --task 2 --source_dir /path/to/BraTS
  python scripts/prepare_plan_b_data.py --task 3 --source_dir /path/to/IXI
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# Official DOI (250 training cases): zenodo.org/records/7153326
ISLES_ZENODO = "https://zenodo.org/records/7153326/files/ISLES-2022.zip?download=1"
IXI_T1_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
IXI_T2_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar"
IXI_DEMO_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls"

TASK_NAMES = {1: "Task001_FOMO1", 2: "Task002_FOMO2", 3: "Task003_FOMO3"}
# Channel count per task (matches FewShotFOMODataset TASK_CONFIGS):
# Task 1: 4 (DWI, FLAIR, ADC, pad); Task 2: 3 (T2, FLAIR, T1ce); Task 3: 2 (T1, T2)
TARGET_SPACING = (1.0, 1.0, 1.0)
DEFAULT_DATA_ROOT = "data/preprocessed"
DEFAULT_N_SHOTS = [16, 32, 64]


def _ensure_nibabel():
    if not HAS_NIBABEL:
        print("ERROR: nibabel is required. Install with: pip install nibabel")
        sys.exit(1)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)
    return dest


def _load_nifti(path: str) -> tuple:
    _ensure_nibabel()
    img = nib.load(path)
    return np.asarray(img.get_fdata(), dtype=np.float32), img.affine


def _resample_volume(vol: np.ndarray, current_spacing: tuple, target_spacing: tuple) -> np.ndarray:
    """Resample spatial dims. vol can be 3D or 4D (D,H,W) or (D,H,W,C); channel dim gets factor 1."""
    from scipy.ndimage import zoom
    n_spatial = min(3, len(current_spacing))
    factors = [current_spacing[i] / target_spacing[i] for i in range(n_spatial)]
    if vol.ndim == 4:
        factors = factors + [1.0]  # do not zoom channel
    return zoom(vol, factors, order=1)


def _get_spacing_from_affine(affine: np.ndarray) -> tuple:
    return tuple(float(np.abs(affine[i, i])) for i in range(3))


def _get_spacing_from_nifti(path: str) -> tuple:
    """Get voxel spacing from NIfTI header (more reliable than affine diagonal for IXI)."""
    _ensure_nibabel()
    img = nib.load(path)
    return tuple(float(x) for x in img.header.get_zooms()[:3])


def _align_vol_to_ref(data: np.ndarray, ref_shape: tuple) -> np.ndarray:
    """Align volume to ref_shape (D,H,W). Squeeze 4D->3D if needed."""
    from scipy.ndimage import zoom
    data = np.asarray(data, dtype=np.float32)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data.squeeze(-1)
    if data.ndim == 4:
        data = data[:, :, :, 0]
    ref_shape = tuple(ref_shape)[:3]
    if data.shape == ref_shape:
        return data
    z = [ref_shape[i] / data.shape[i] for i in range(3)]
    return zoom(data, z, order=1)


def _zscore_normalize(vol: np.ndarray) -> np.ndarray:
    """Per-channel z-score (masked by >0) to align with FOMO pretraining."""
    out = vol.copy()
    for c in range(vol.shape[0]):
        ch = vol[c]
        mask = ch > 0
        if mask.sum() > 0:
            mean, std = ch[mask].mean(), ch[mask].std()
            out[c] = np.where(mask, (ch - mean) / (std + 1e-8), ch)
    return out.astype(np.float32)


def _find_isles_case_files(root: Path) -> list:
    cases = []
    for sub_dir in sorted(root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        flair = list(sub_dir.rglob("*FLAIR*.nii*")) + list(sub_dir.rglob("*flair*.nii*"))
        dwi = list(sub_dir.rglob("*DWI*.nii*")) + list(sub_dir.rglob("*dwi*.nii*"))
        adc = list(sub_dir.rglob("*ADC*.nii*")) + list(sub_dir.rglob("*adc*.nii*"))
        mask = list(sub_dir.rglob("*mask*.nii*")) + list(sub_dir.rglob("*lesion*.nii*"))
        mask += [m for m in sub_dir.rglob("*.nii*") if "label" in str(m).lower() or "derivatives" in str(m)]
        if flair and dwi and adc:
            cases.append({
                "flair": flair[0], "dwi": dwi[0], "adc": adc[0],
                "mask": mask[0] if mask else None,
                "id": sub_dir.name
            })
    if not cases:
        for case_dir in sorted(root.iterdir()):
            if not case_dir.is_dir():
                continue
            files = list(case_dir.glob("*.nii*"))
            flair = [f for f in files if "flair" in f.name.lower()]
            dwi = [f for f in files if "dwi" in f.name.lower() and "adc" not in f.name.lower()]
            adc = [f for f in files if "adc" in f.name.lower()]
            mask = [f for f in files if "mask" in f.name.lower() or "lesion" in f.name.lower() or "seg" in f.name.lower()]
            if flair and dwi and adc:
                cases.append({"flair": flair[0], "dwi": dwi[0], "adc": adc[0], "mask": mask[0] if mask else None, "id": case_dir.name})
    return cases


def prepare_task1(source_dir: Path, out_dir: Path) -> int:
    _ensure_nibabel()
    task_dir = out_dir / TASK_NAMES[1]
    task_dir.mkdir(parents=True, exist_ok=True)
    cases = _find_isles_case_files(source_dir)
    if not cases:
        print("WARNING: No ISLES cases found.")
        return 0
    n = 0
    for c in cases:
        try:
            dwi_data, dwi_aff = _load_nifti(str(c["dwi"]))
            flair_data, _ = _load_nifti(str(c["flair"]))
            adc_data, _ = _load_nifti(str(c["adc"]))
            while dwi_data.ndim > 3 and dwi_data.shape[-1] == 1:
                dwi_data = dwi_data.squeeze(-1)
            if dwi_data.ndim == 4:
                dwi_data = dwi_data[:, :, :, 0]
            ref_shape = dwi_data.shape[:3]
            ref_spacing = _get_spacing_from_affine(dwi_aff)
            flair_data = _align_vol_to_ref(flair_data, ref_shape)
            adc_data = _align_vol_to_ref(adc_data, ref_shape)
            vol = np.stack([dwi_data, flair_data, adc_data, np.zeros_like(dwi_data)], axis=0)
            if TARGET_SPACING != ref_spacing:
                vol4d = np.moveaxis(vol, 0, -1)
                vol4d = _resample_volume(vol4d, ref_spacing + (1,), TARGET_SPACING + (1,))
                vol = np.moveaxis(vol4d, -1, 0)
            if c["mask"] is not None:
                mask_data, _ = _load_nifti(str(c["mask"]))
                mask_data = _align_vol_to_ref(mask_data, vol.shape[1:])
                label = 1 if np.any(mask_data > 0.5) else 0
            else:
                label = 1
            vol = _zscore_normalize(vol)
            safe_id = re.sub(r"[^\w\-]", "_", c["id"])[:64]
            base = task_dir / f"sample_{safe_id}"
            np.save(str(base) + ".npy", vol.astype(np.float32))
            np.savetxt(str(base) + ".txt", [label], fmt="%d")
            n += 1
        except Exception as e:
            print(f"Skip {c['id']}: {e}")
    return n


def _find_brats_case_files(root: Path) -> list:
    """Support BraTS 2023 (t1n,t1c,t2w,t2f) and BraTS Meningioma (t1,t1ce,t2,flair) naming."""
    cases = []
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        files = list(item.rglob("*.nii*"))
        t1 = [f for f in files if any(x in f.name.lower() for x in ["-t1n", "_t1.nii", "-t1-"]) and "t1ce" not in f.name.lower() and "t1c" not in f.name.lower()]
        t1ce = [f for f in files if any(x in f.name.lower() for x in ["t1ce", "-t1c", "_t1ce"])]
        t2 = [f for f in files if any(x in f.name.lower() for x in ["-t2w", "_t2.nii", "-t2-"]) and "flair" not in f.name.lower() and "t2f" not in f.name.lower()]
        flair = [f for f in files if any(x in f.name.lower() for x in ["flair", "-t2f", "t2flair"])]
        seg = [f for f in files if "seg" in f.name.lower()]
        if not (t2 and flair):
            continue
        ch3 = t1ce[0] if t1ce else (t1[0] if t1 else t2[0])
        cases.append({"t2": t2[0], "flair": flair[0], "ch3": ch3, "seg": seg[0] if seg else None, "id": item.name})
    return cases


def prepare_task2(source_dir: Path, out_dir: Path) -> int:
    _ensure_nibabel()
    task_dir = out_dir / TASK_NAMES[2]
    task_dir.mkdir(parents=True, exist_ok=True)
    cases = _find_brats_case_files(source_dir)
    if not cases:
        print("WARNING: No BraTS cases found.")
        return 0
    n = 0
    for c in cases:
        try:
            t2_data, t2_aff = _load_nifti(str(c["t2"]))
            flair_data = _load_nifti(str(c["flair"]))[0]
            ch3_data = _load_nifti(str(c["ch3"]))[0]
            ref_shape = tuple(t2_data.shape)[:3]
            ref_spacing = _get_spacing_from_affine(t2_aff)
            vol = np.stack([_align_vol_to_ref(t2_data, ref_shape), _align_vol_to_ref(flair_data, ref_shape), _align_vol_to_ref(ch3_data, ref_shape)], axis=0)
            if TARGET_SPACING != ref_spacing:
                vol4d = np.moveaxis(vol, 0, -1)
                vol4d = _resample_volume(vol4d, ref_spacing + (1,), TARGET_SPACING + (1,))
                vol = np.moveaxis(vol4d, -1, 0)
            if c["seg"] is not None:
                seg_data = _load_nifti(str(c["seg"]))[0]
                seg_data = _align_vol_to_ref(seg_data, vol.shape[1:])
                seg_bin = (seg_data > 0.5).astype(np.float32)[np.newaxis, ...]
            else:
                seg_bin = np.zeros((1,) + vol.shape[1:], dtype=np.float32)
            vol = _zscore_normalize(vol)
            safe_id = re.sub(r"[^\w\-]", "_", c["id"])[:64]
            base = task_dir / f"sample_{safe_id}"
            np.save(str(base) + ".npy", vol.astype(np.float32))
            np.save(str(base) + "_seg.npy", seg_bin.astype(np.float32))
            n += 1
        except Exception as e:
            print(f"Skip {c['id']}: {e}")
    return n


def _load_ixi_demographics(path: Path) -> dict:
    mapping = {}
    try:
        import pandas as pd
        df = pd.read_excel(path, engine="xlrd") if path.suffix.lower() == ".xls" else pd.read_excel(path)
        id_col = next((c for c in df.columns if "id" in c.lower() or "IXI" in str(c).upper()), df.columns[0])
        age_col = next((c for c in df.columns if "age" in c.lower()), df.columns[1] if len(df.columns) > 1 else df.columns[0])
        for _, row in df.iterrows():
            sid = str(row[id_col]).strip()
            try:
                mapping[sid] = float(row[age_col])
            except (ValueError, TypeError):
                mapping[sid] = 50.0
    except Exception as e:
        print(f"Demographics warning: {e}")
    return mapping


def _find_ixi_case_files(t1_dir: Path, t2_dir: Path) -> list:
    def _ixi_id(path):
        m = re.search(r"(\d{2,4})", path.stem)
        return m.group(1) if m else path.stem

    t1_by_id = {}
    for f in t1_dir.glob("*.nii*"):
        if "T1" in f.name.upper():
            t1_by_id[_ixi_id(f)] = f
    t2_by_id = {}
    for f in t2_dir.glob("*.nii*"):
        if "T2" in f.name.upper():
            t2_by_id[_ixi_id(f)] = f
    common = set(t1_by_id.keys()) & set(t2_by_id.keys())
    return [{"id": f"IXI-{sid}", "t1": t1_by_id[sid], "t2": t2_by_id[sid]} for sid in sorted(common)]


def prepare_task3(source_dir: Path, out_dir: Path, demo_path: Path = None) -> int:
    _ensure_nibabel()
    task_dir = out_dir / TASK_NAMES[3]
    task_dir.mkdir(parents=True, exist_ok=True)
    t1_dir = source_dir / "IXI-T1"
    t2_dir = source_dir / "IXI-T2"
    if not t1_dir.exists() or len(list(t1_dir.glob("*T1*.nii*"))) == 0:
        t1_dir = source_dir
    if not t2_dir.exists() or len(list(t2_dir.glob("*T2*.nii*"))) == 0:
        t2_dir = source_dir
    cases = _find_ixi_case_files(t1_dir, t2_dir)
    if not cases:
        print("WARNING: No IXI T1+T2 pairs found.")
        return 0
    demo = _load_ixi_demographics(demo_path) if demo_path and demo_path.exists() else {}
    n = 0
    for c in cases:
        try:
            t1_data, t1_aff = _load_nifti(str(c["t1"]))
            t2_data, _ = _load_nifti(str(c["t2"]))
            ref_shape = tuple(t1_data.shape)[:3]
            ref_spacing = _get_spacing_from_nifti(str(c["t1"]))  # IXI affine diagonal wrong; use header zooms
            t2_data = _align_vol_to_ref(t2_data, ref_shape)
            vol = np.stack([t1_data.astype(np.float32), t2_data.astype(np.float32)], axis=0)
            if TARGET_SPACING != ref_spacing:
                vol4d = np.moveaxis(vol, 0, -1)
                vol4d = _resample_volume(vol4d, ref_spacing + (1,), TARGET_SPACING + (1,))
                vol = np.moveaxis(vol4d, -1, 0)
            vol = _zscore_normalize(vol)
            sid = c["id"]
            age = demo.get(sid) or demo.get(sid.replace("IXI-", "")) or demo.get(sid.split("-")[-1]) or 50.0
            if not isinstance(age, (int, float)):
                age = 50.0
            safe_id = re.sub(r"[^\w\-]", "_", c["id"])[:64]
            base = task_dir / f"sample_{safe_id}"
            np.save(str(base) + ".npy", vol.astype(np.float32))
            np.savetxt(str(base) + ".txt", [float(age)], fmt="%.2f")
            n += 1
        except Exception as e:
            print(f"Skip {c['id']}: {e}")
    return n


def add_ixi_healthy_to_task1(ixi_source: Path, out_dir: Path, max_samples: int = 250) -> int:
    """
    Add IXI healthy subjects as Task 1 negatives (label=0).
    IXI has T1+T2; we pad to 4ch as [T1, T2, zeros, zeros] for Task 1 format.
    """
    _ensure_nibabel()
    task_dir = out_dir / TASK_NAMES[1]
    task_dir.mkdir(parents=True, exist_ok=True)
    t1_dir = ixi_source if not (ixi_source / "IXI-T1").exists() else ixi_source / "IXI-T1"
    t2_dir = ixi_source if not (ixi_source / "IXI-T2").exists() else ixi_source / "IXI-T2"
    if not t1_dir.exists():
        t1_dir = ixi_source
    if not t2_dir.exists():
        t2_dir = ixi_source
    cases = _find_ixi_case_files(t1_dir, t2_dir)
    if not cases:
        print("WARNING: No IXI pairs for healthy negatives.")
        return 0
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(cases))[:max_samples]
    cases = [cases[i] for i in perm]
    n = 0
    for c in cases:
        try:
            t1_data, t1_aff = _load_nifti(str(c["t1"]))
            t2_data, _ = _load_nifti(str(c["t2"]))
            ref_shape = tuple(t1_data.shape)[:3]
            ref_spacing = _get_spacing_from_nifti(str(c["t1"]))
            t2_data = _align_vol_to_ref(t2_data, ref_shape)
            vol = np.stack([t1_data.astype(np.float32), t2_data.astype(np.float32), np.zeros_like(t1_data), np.zeros_like(t1_data)], axis=0)
            if TARGET_SPACING != ref_spacing:
                vol4d = np.moveaxis(vol, 0, -1)
                vol4d = _resample_volume(vol4d, ref_spacing + (1,), TARGET_SPACING + (1,))
                vol = np.moveaxis(vol4d, -1, 0)
            vol = _zscore_normalize(vol)
            safe_id = "ixi_healthy_" + re.sub(r"[^\w\-]", "_", c["id"])[:48]
            base = task_dir / f"sample_{safe_id}"
            np.save(str(base) + ".npy", vol.astype(np.float32))
            np.savetxt(str(base) + ".txt", [0], fmt="%d")
            n += 1
        except Exception as e:
            print(f"Skip IXI {c['id']}: {e}")
    return n


def create_fewshot_split(
    task_dir: Path,
    n_shots: list = None,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Create few-shot splits from converted data.
    - fewshot_{k}/: k samples for train; copies .npy, .txt, _seg.npy
    - splits_fewshot_{k}.json: {"train": [...], "val": [...]} for reproducibility
    - Val = remaining samples (used for reporting Dice/AUROC)
    """
    import json
    n_shots = n_shots or DEFAULT_N_SHOTS
    samples = sorted([f for f in task_dir.glob("sample_*.npy") if "_seg" not in f.name])
    if not samples:
        return {}
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(samples))
    samples = [samples[i] for i in perm]
    out = {}
    for k in n_shots:
        k = min(k, len(samples))
        split_dir = task_dir.parent / f"{task_dir.name}_fewshot{k}"
        split_dir.mkdir(exist_ok=True)
        train_ids = list(range(k))
        val_ids = list(range(k, len(samples)))
        for idx in train_ids:
            src = samples[idx]
            stem = src.stem
            shutil.copy(src, split_dir / src.name)
            for suff in [".txt", "_seg.npy"]:
                lab = src.with_name(stem + suff)
                if lab.exists():
                    shutil.copy(lab, split_dir / lab.name)
        train_stems = [samples[i].stem for i in train_ids]
        val_stems = [samples[i].stem for i in val_ids]
        out[f"fewshot_{k}"] = {"train": train_stems, "val": val_stems}
        split_json = task_dir.parent / f"splits_{task_dir.name}_fewshot{k}.json"
        with open(split_json, "w") as f:
            json.dump({task_dir.name: out[f"fewshot_{k}"]}, f, indent=2)
    return out


def download_isles(dest: Path) -> Path:
    zip_path = dest / "ISLES-2022.zip"
    if not zip_path.exists():
        print("Downloading ISLES 2022...")
        _download(ISLES_ZENODO, zip_path)
    ext_dir = dest / "ISLES-2022"
    if not ext_dir.exists():
        print("Extracting ISLES 2022...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
    return ext_dir


def download_ixi(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    has_t1 = len(list(dest.glob("*T1*.nii*"))) > 0
    has_t2 = len(list(dest.glob("*T2*.nii*"))) > 0
    for name, url in [("IXI-T1", IXI_T1_URL), ("IXI-T2", IXI_T2_URL)]:
        tar_path = dest / f"{name}.tar"
        if not tar_path.exists():
            print(f"Downloading {name}...")
            _download(url, tar_path)
        need_extract = ("T1" in name and not has_t1) or ("T2" in name and not has_t2)
        if need_extract:
            print(f"Extracting {name}...")
            with tarfile.open(tar_path) as t:
                t.extractall(dest)
    demo_path = dest / "IXI.xls"
    if not demo_path.exists():
        try:
            _download(IXI_DEMO_URL, demo_path)
        except Exception:
            pass
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=[1, 2, 3])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--source_dir", type=str)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work_dir", type=str, default="data/plan_b_raw")
    parser.add_argument("--create_fewshot", action="store_true",
                        help="Create fewshot_16/32/64 dirs and splits JSON after conversion")
    parser.add_argument("--n_shots", type=int, nargs="+", default=DEFAULT_N_SHOTS)
    parser.add_argument("--add_ixi_healthy", action="store_true",
                        help="Add IXI healthy subjects as Task 1 negatives (needs IXI in work_dir/ixi)")
    parser.add_argument("--n_healthy", type=int, default=250, help="Max IXI healthy samples for Task 1")
    args = parser.parse_args()

    tasks = [1, 2, 3] if (args.task is None or args.all) else [args.task]
    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)

    if args.download:
        if 2 in tasks:
            print("Task 2 (BraTS-Meningioma) requires manual download from Synapse:")
            print("  https://www.synapse.org/#!Synapse:syn51514105")
            print("  (Sign in, accept DUA, then download.)")
            print("  Then run: --task 2 --source_dir /path/to/downloaded/data")
        if 1 in tasks:
            src1 = download_isles(work_dir / "isles")
            n = prepare_task1(src1, out_dir)
            print(f"Task 1: converted {n} cases")
        if 3 in tasks:
            src3 = download_ixi(work_dir / "ixi")
            n = prepare_task3(src3, out_dir, src3 / "IXI.xls")
            print(f"Task 3: converted {n} cases")
        if args.add_ixi_healthy and 1 in tasks:
            ixi_path = work_dir / "ixi"
            if not ixi_path.exists() or len(list(ixi_path.glob("*T1*.nii*"))) == 0:
                ixi_path = download_ixi(ixi_path)
            n = add_ixi_healthy_to_task1(ixi_path, out_dir, max_samples=args.n_healthy)
            print(f"Task 1: added {n} IXI healthy (negatives)")

    if args.source_dir:
        src = Path(args.source_dir)
        if not src.exists():
            print(f"ERROR: source_dir does not exist: {src}")
            sys.exit(1)
        for t in tasks:
            if t == 1:
                n = prepare_task1(src, out_dir)
                print(f"Task 1: converted {n} cases")
            elif t == 2:
                n = prepare_task2(src, out_dir)
                print(f"Task 2: converted {n} cases")
            elif t == 3:
                n = prepare_task3(src, out_dir, src / "IXI.xls")
                print(f"Task 3: converted {n} cases")

    if args.add_ixi_healthy and 1 in tasks:
        ixi_path = work_dir / "ixi"
        alt = Path(args.source_dir) if args.source_dir else None
        if alt and (alt / "IXI-T1").exists():
            ixi_path = alt
        elif alt and len(list(alt.glob("*T1*.nii*"))) > 0:
            ixi_path = alt
        if ixi_path.exists() and len(list(ixi_path.glob("*T1*.nii*"))) > 0:
            n = add_ixi_healthy_to_task1(ixi_path, out_dir, max_samples=args.n_healthy)
            print(f"Task 1: added {n} IXI healthy (negatives)")

    if args.create_fewshot:
        for t in tasks:
            task_dir = out_dir / TASK_NAMES[t]
            if task_dir.exists():
                create_fewshot_split(task_dir, n_shots=args.n_shots)
                print(f"Task {t}: created fewshot splits for {args.n_shots}")

    if not args.download and not args.source_dir and not args.create_fewshot:
        print("Use --download and/or --source_dir and/or --create_fewshot. See --help.")
        sys.exit(0)


if __name__ == "__main__":
    main()
