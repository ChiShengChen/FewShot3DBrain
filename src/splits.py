"""
Subject-grouped, three-way splits for the few-shot continual learning benchmark.

Two properties the original two-way split did not have:

1. **A held-out test set.** Model selection previously ran on the same subset that was
   reported, so every headline number was a best-of-100-epochs maximum on the scoring
   set. Train / val / test are now disjoint: val picks the epoch, test is reported once.

2. **Subject-level grouping.** BraTS ships longitudinal timepoints (138 subjects have
   more than one), so splitting on files leaks the same patient across train and test.
   Splits are built over subject groups, never individual volumes.

Splits are written to JSON so that every method and every baseline is scored on exactly
the same cases.
"""
import json
import os
import re
from typing import Dict, List, Optional

import numpy as np

from .data import TASK_CONFIGS, list_case_files


def subject_id(path: str) -> str:
    """Group key for a case path. Falls back to the case name when no pattern matches."""
    name = os.path.basename(path)
    name = name[len("sample_"):] if name.startswith("sample_") else name

    # BraTS: BraTS-GLI-00001-000 / -001 are timepoints of one subject.
    m = re.match(r"(BraTS-[A-Za-z]+-\d+)-\d+$", name)
    if m:
        return m.group(1)
    # IXI (brain age) and IXI healthy controls reused as Task 1 negatives.
    m = re.match(r"(?:ixi_healthy_)?(IXI-\d+)", name)
    if m:
        return m.group(1)
    # ISLES 2022: sub-strokecase0001_ses-0001 -> one subject, one session.
    m = re.match(r"(sub-strokecase\d+)", name)
    if m:
        return m.group(1)
    return name


def make_splits(
    data_dir: str,
    task_id: int,
    n_shot: int,
    seed: int = 42,
    val_subjects: int = 60,
    max_test_subjects: Optional[int] = 250,
) -> Dict[str, List[str]]:
    """Build a subject-disjoint train/val/test split.

    train gets exactly ``n_shot`` volumes (the few-shot budget), val is used only for
    epoch selection, and test is the reported set. ``max_test_subjects`` caps evaluation
    cost; it is applied after shuffling so the cap does not bias the sample.
    """
    files = list_case_files(data_dir, task_id)
    if not files:
        raise FileNotFoundError(f"No usable cases for task {task_id} under {data_dir}")

    by_subject: Dict[str, List[str]] = {}
    for f in files:
        by_subject.setdefault(subject_id(f), []).append(f)
    subjects = sorted(by_subject)

    rng = np.random.default_rng(seed)
    order = [subjects[i] for i in rng.permutation(len(subjects))]

    # Train: consume whole subjects until the shot budget is met, then truncate to
    # exactly n_shot volumes so the few-shot budget is honoured to the case.
    train: List[str] = []
    cursor = 0
    while cursor < len(order) and len(train) < n_shot:
        train.extend(sorted(by_subject[order[cursor]]))
        cursor += 1
    train = train[:n_shot]

    val: List[str] = []
    val_start = cursor
    while cursor < len(order) and (cursor - val_start) < val_subjects:
        val.extend(sorted(by_subject[order[cursor]]))
        cursor += 1

    test_subjects = order[cursor:]
    if max_test_subjects is not None:
        test_subjects = test_subjects[:max_test_subjects]
    test = [f for s in test_subjects for f in sorted(by_subject[s])]

    assert not (set(train) & set(val)), "train/val overlap"
    assert not (set(train) & set(test)), "train/test overlap"
    assert not (set(val) & set(test)), "val/test overlap"
    train_subj = {subject_id(f) for f in train}
    assert not (train_subj & {subject_id(f) for f in test}), "subject leaks train->test"

    return {"train": train, "val": val, "test": test}


def split_path(split_dir: str, task_id: int, n_shot: int, seed: int) -> str:
    name = TASK_CONFIGS[task_id]["name"]
    return os.path.join(split_dir, f"{name}_n{n_shot}_seed{seed}.json")


def load_or_make_splits(
    data_dir: str,
    task_id: int,
    n_shot: int,
    seed: int = 42,
    split_dir: str = "data/splits_v2",
    **kwargs,
) -> Dict[str, List[str]]:
    """Return a cached split, creating and persisting it on first use."""
    os.makedirs(split_dir, exist_ok=True)
    path = split_path(split_dir, task_id, n_shot, seed)
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)["splits"]

    splits = make_splits(data_dir, task_id, n_shot, seed=seed, **kwargs)
    payload = {
        "task_id": task_id,
        "task_name": TASK_CONFIGS[task_id]["name"],
        "n_shot": n_shot,
        "seed": seed,
        "counts": {k: len(v) for k, v in splits.items()},
        "n_subjects": {k: len({subject_id(f) for f in v}) for k, v in splits.items()},
        "splits": splits,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return splits
