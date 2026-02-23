#!/usr/bin/env python3
"""Debug IXI raw data shape."""
import nibabel as nib
from pathlib import Path

ixi_root = Path("data/plan_b_raw/ixi")
t1_files = list(ixi_root.rglob("*T1*.nii*"))
if not t1_files:
    print(f"No IXI T1 files found under {ixi_root}")
    exit(1)
f = t1_files[0]
img = nib.load(str(f))
print(f"File: {f}")
print(f"Raw IXI T1 shape: {img.shape}")
print(f"Affine:\n{img.affine}")
