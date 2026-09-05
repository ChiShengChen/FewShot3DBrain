"""
Guards for the data-loading and split bugs that invalidated the first round of results.

Run with:  python -m pytest tests/test_data_integrity.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import FewShotFOMODataset, list_case_files  # noqa: E402
from src.splits import load_or_make_splits, subject_id  # noqa: E402

DATA_DIR = "data/preprocessed"
have_data = os.path.isdir(DATA_DIR)
needs_data = pytest.mark.skipif(not have_data, reason="preprocessed data not available")


# ---------------------------------------------------------------- label plumbing

def _make_scalar_case(tmp_path, task_type, values):
    files = []
    for i, v in enumerate(values):
        base = str(tmp_path / f"sample_{i:03d}")
        np.save(base + ".npy", np.random.rand(2, 80, 80, 80).astype(np.float32))
        np.savetxt(base + ".txt", [v])
        files.append(base)
    return files


def test_regression_labels_survive_cropping(tmp_path):
    """The crop step must not clobber scalar labels (it silently forced every age to 50.0)."""
    ages = [23.5, 41.0, 67.25, 80.0]
    files = _make_scalar_case(tmp_path, "regression", ages)
    ds = FewShotFOMODataset(files, patch_size=(64, 64, 64), task_type="regression")
    got = [float(ds[i]["label"]) for i in range(len(files))]
    assert got == pytest.approx(ages), f"expected {ages}, got {got}"
    assert len(set(got)) > 1, "all regression labels collapsed to a constant"


def test_classification_labels_survive_cropping(tmp_path):
    labels = [0, 1, 1, 0]
    files = _make_scalar_case(tmp_path, "classification", labels)
    ds = FewShotFOMODataset(files, patch_size=(64, 64, 64), task_type="classification")
    got = [int(ds[i]["label"]) for i in range(len(files))]
    assert got == labels
    assert len(set(got)) > 1, "all classification labels collapsed to a constant"


def test_segmentation_label_is_cropped_with_volume(tmp_path):
    base = str(tmp_path / "sample_000")
    np.save(base + ".npy", np.random.rand(3, 80, 80, 80).astype(np.float32))
    np.save(base + "_seg.npy", (np.random.rand(80, 80, 80) > 0.5).astype(np.uint8))
    ds = FewShotFOMODataset([base], patch_size=(64, 64, 64), task_type="segmentation")
    item = ds[0]
    assert item["image"].shape[1:] == item["label"].shape[1:] == (64, 64, 64)


def test_nonfinite_scalar_labels_are_excluded(tmp_path):
    task_dir = tmp_path / "Task003_FOMO3"
    task_dir.mkdir()
    for i, v in enumerate([30.0, float("nan"), 55.0]):
        base = str(task_dir / f"sample_IXI-{i:03d}")
        np.save(base + ".npy", np.random.rand(2, 80, 80, 80).astype(np.float32))
        np.savetxt(base + ".txt", [v])
    assert len(list_case_files(str(tmp_path), 3)) == 2
    assert len(list_case_files(str(tmp_path), 3, require_usable_label=False)) == 3


# ---------------------------------------------------------------- splits

def test_subject_id_groups_brats_timepoints():
    assert subject_id("x/sample_BraTS-GLI-00001-000") == "BraTS-GLI-00001"
    assert subject_id("x/sample_BraTS-GLI-00001-001") == "BraTS-GLI-00001"
    assert subject_id("x/sample_IXI-002") == "IXI-002"
    assert subject_id("x/sample_ixi_healthy_IXI-002") == "IXI-002"


@needs_data
@pytest.mark.parametrize("task_id", [2, 3])
def test_splits_are_disjoint_at_subject_level(task_id, tmp_path):
    s = load_or_make_splits(DATA_DIR, task_id, 32, seed=42, split_dir=str(tmp_path))
    subj = {k: {subject_id(f) for f in v} for k, v in s.items()}
    assert not (subj["train"] & subj["val"])
    assert not (subj["train"] & subj["test"])
    assert not (subj["val"] & subj["test"])
    assert len(s["train"]) == 32
    assert len(s["test"]) > 0


@needs_data
def test_splits_are_deterministic(tmp_path):
    a = load_or_make_splits(DATA_DIR, 3, 32, seed=42, split_dir=str(tmp_path / "a"))
    b = load_or_make_splits(DATA_DIR, 3, 32, seed=42, split_dir=str(tmp_path / "b"))
    assert a == b


# ---------------------------------------------------------------- prepared data

@needs_data
def test_ixi_ages_are_real_not_placeholders():
    """A working IXI join yields a wide age distribution; the broken one gave a spike at 50."""
    files = list_case_files(DATA_DIR, 3)
    ages = np.array([float(np.loadtxt(f + ".txt")) for f in files])
    assert np.isfinite(ages).all()
    assert ages.std() > 10.0, f"age std {ages.std():.2f} is too small to be real"
    assert (ages == 50.0).mean() < 0.05, "suspicious spike of placeholder ages at 50.0"


@needs_data
def test_task1_volumes_are_not_degenerate():
    """IXI healthy negatives were written as collapsed volumes, making Task 1 trivially separable."""
    files = list_case_files(DATA_DIR, 1)
    if not files:
        pytest.skip("Task 1 not prepared")
    bad = [f for f in files if min(np.load(f + ".npy", mmap_mode="r").shape[1:]) < 32]
    assert not bad, f"{len(bad)}/{len(files)} Task 1 volumes are degenerate, e.g. {bad[:3]}"
