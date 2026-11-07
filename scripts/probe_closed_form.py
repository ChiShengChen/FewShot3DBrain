#!/usr/bin/env python3
"""
Closed-form linear probe on frozen pooled encoder features (reviewer control, JBHI revision 2).

The gradient-trained linear probe (Sequential Linear: a 513-parameter head trained with AdamW at
lr 1e-3 for 100 epochs on raw pooled features) returns a constant on every whole-head task. That
could be a property of the features or an artefact of optimising a tiny head on un-standardised
inputs whose between-subject spread is ~1% of their magnitude. This probe removes the optimiser
from the question: it z-scores the pooled features with training-split statistics and fits ridge
regression (regression tasks) or L2-regularised logistic regression (classification tasks) in
closed form / to convergence, choosing the regularisation strength on the validation split.

  python scripts/probe_closed_form.py --tasks 5 6 9 10 11 --seeds 42 43 44 45 46
  python scripts/probe_closed_form.py --tasks 5 6 9 10 --seeds 42 --full_train
  python scripts/probe_closed_form.py --tasks 5 6 9 10 11 --pretrained_path ''   # random features

Writes outputs/benchmark_v2/n{32|full}/ridge_probe_t{T}[_randinit]/seed{S}/results.json and
task{T}_per_case.json in the same format as scripts/run_benchmark_v2.py.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))

from src.data import TASK_CONFIGS, list_case_files  # noqa: E402
from src.eval import _load_scalar, _load_volume  # noqa: E402
from src.models import TaskModel  # noqa: E402
from src.splits import load_or_make_splits  # noqa: E402

DEFAULT_WEIGHTS = str(ROOT / "weights/fomo25_mmunetvae_pretrained.ckpt")


@torch.no_grad()
def extract_features(task_id, files, pretrained, patch, device, seed=0):
    """Pooled deepest-stage encoder feature (512-d) of each case, one forward pass per case."""
    torch.manual_seed(seed)  # fixes the random-init weights when pretrained == ''
    cfg = TASK_CONFIGS[task_id]
    model = TaskModel(task_id, cfg["modalities"], (patch,) * 3, pretrained_path=pretrained or None,
                      model_name="unet_b", lora_r=0, lora_decoder=False).to(device).eval()
    feats = {}
    for f in files:
        vol = _load_volume(f)[: cfg["modalities"]]
        c, *sp = vol.shape
        pad = [max(0, patch - s) for s in sp]
        if any(pad):
            vol = np.pad(vol, ((0, 0), (0, pad[0]), (0, pad[1]), (0, pad[2])))
        z, y, x = [(s - patch) // 2 for s in vol.shape[1:]]
        vol = vol[:, z:z + patch, y:y + patch, x:x + patch]
        enc = model.backbone(torch.from_numpy(np.ascontiguousarray(vol))[None].to(device))
        feats[os.path.basename(f)] = enc[-1].mean(dim=(2, 3, 4)).squeeze(0).float().cpu().numpy()
    return feats


def regression_metrics(p, gts):
    resid = p - gts
    ss_res, ss_tot = float((resid ** 2).sum()), float(((gts - gts.mean()) ** 2).sum())
    return {
        "n": int(len(p)), "mae": float(np.abs(resid).mean()), "rmse": float(np.sqrt((resid ** 2).mean())),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson_r": float(np.corrcoef(p, gts)[0, 1]) if p.std() > 0 and gts.std() > 0 else float("nan"),
        "bias": float(resid.mean()), "pred_std": float(p.std()), "gt_std": float(gts.std()),
        "calibration_slope": float(np.polyfit(gts, p, 1)[0]) if gts.std() > 0 else float("nan"),
    }


def classification_metrics(score, y):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    return {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(y, (score > 0.5).astype(int))),
        "pos_rate": float(y.mean()),
    }


def fit_probe(task_type, Xtr, ytr, Xva, yva):
    """Choose the regularisation strength on validation, refit on train. Returns (predict, info)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    if task_type == "regression":
        best = None
        for a in np.logspace(-3, 5, 17):
            m = Ridge(alpha=a).fit(Xtr, ytr)
            mae = float(np.abs(m.predict(Xva) - yva).mean())
            if best is None or mae < best[0] - 1e-12:
                best = (mae, a, m)
        return best[2].predict, {"alpha": float(best[1]), "val_mae": best[0]}
    from sklearn.metrics import roc_auc_score
    best = None
    for C in np.logspace(-4, 3, 15):
        m = LogisticRegression(C=C, max_iter=10000).fit(Xtr, ytr.astype(int))
        s = m.predict_proba(Xva)[:, 1]
        auc = float(roc_auc_score(yva, s)) if len(np.unique(yva)) > 1 else 0.5
        if best is None or auc > best[0] + 1e-12:
            best = (auc, C, m)
    return (lambda X, m=best[2]: m.predict_proba(X)[:, 1]), {"C": float(best[1]), "val_auc": best[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="+", default=[5, 6, 9, 10, 11])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n_shot", type=int, default=32)
    ap.add_argument("--full_train", action="store_true")
    ap.add_argument("--patch_size", type=int, default=96)
    ap.add_argument("--pretrained_path", default=DEFAULT_WEIGHTS, help="'' = randomly initialised encoder")
    ap.add_argument("--data_dir", default=str(ROOT / "data/preprocessed"))
    ap.add_argument("--split_dir", default=str(ROOT / "data/splits_v3"))
    ap.add_argument("--save_dir", default=str(ROOT / "outputs/benchmark_v2"))
    ap.add_argument("--cache_dir", default=str(ROOT / "outputs/benchmark_v2/probe_feature_cache"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = "" if args.pretrained_path else "_randinit"
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        cfg = TASK_CONFIGS[task]
        ttype = cfg["task_type"]
        cache = Path(args.cache_dir) / f"task{task}{tag}_p{args.patch_size}.npz"
        files_all = list_case_files(args.data_dir, task)
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            feats = dict(zip(z["names"].tolist(), z["X"]))
        else:
            t0 = time.perf_counter()
            feats = extract_features(task, files_all, args.pretrained_path, args.patch_size, device)
            np.savez(cache, names=np.array(list(feats)), X=np.stack(list(feats.values())))
            print(f"task {task}: extracted {len(feats)} features in {time.perf_counter() - t0:.0f}s", flush=True)

        for seed in args.seeds:
            sp = load_or_make_splits(args.data_dir, task, args.n_shot, seed=seed, split_dir=args.split_dir)
            train = list(sp["train"])
            if args.full_train:
                held = {os.path.basename(f) for f in sp["val"]} | {os.path.basename(f) for f in sp["test"]}
                train = [f for f in files_all if os.path.basename(f) not in held]

            def xy(files):
                X = np.stack([feats[os.path.basename(f)] for f in files])
                y = np.array([_load_scalar(f) for f in files], dtype=np.float64)
                return X, y
            Xtr, ytr = xy(train)
            Xva, yva = xy(sp["val"])
            Xte, yte = xy(sp["test"])
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd < 1e-8] = 1.0
            Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
            predict, info = fit_probe(ttype, Xtr, ytr, Xva, yva)
            p = predict(Xte)
            if ttype == "regression":
                test = regression_metrics(p, yte)
                per_case = [{"case": f, "pred": float(a), "gt": float(b)} for f, a, b in zip(sp["test"], p, yte)]
            else:
                test = classification_metrics(p, yte.astype(int))
                per_case = [{"case": f, "score": float(a), "gt": int(b)} for f, a, b in zip(sp["test"], p, yte)]

            n_tag = "full" if args.full_train else str(args.n_shot)
            out = Path(args.save_dir) / f"n{n_tag}" / f"ridge_probe_t{task}{tag}" / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            n_params = Xtr.shape[1] + 1
            res = {"method": "ridge_probe", "seed": seed, "n_shot": args.n_shot, "tasks": [task],
                   "eval_protocol": {"patch": [args.patch_size] * 3, "features": "pooled encoder_conv4, z-scored on train"},
                   "per_task": {f"task{task}": {
                       "position": 0, "val_selected": info, "test": test,
                       "params": {"trainable": n_params, "total": n_params, "trainable_pct": 100.0},
                       "train_sec": 0.0, "n_train": len(train), "n_val": len(sp["val"]), "n_test": len(sp["test"])}}}
            (out / "results.json").write_text(json.dumps(res, indent=2))
            (out / f"task{task}_per_case.json").write_text(json.dumps(per_case, indent=2))
            key = "mae" if ttype == "regression" else "auc"
            extra = f" R2={test['r2']:.3f} pred_std={test['pred_std']:.2f}" if ttype == "regression" else f" bal_acc={test['balanced_acc']:.3f}"
            print(f"task {task} seed {seed} n_train={len(train)}: {key}={test[key]:.3f}{extra}  ({info})", flush=True)


if __name__ == "__main__":
    main()
