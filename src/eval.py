"""
Deterministic, whole-volume evaluation.

The original protocol scored a model on a single *random* 64^3 crop per case and pooled
every voxel in the split into one global Dice. That makes results irreproducible run to
run, ignores most of each volume, and is not comparable to any published BraTS number.

Here:
  * segmentation runs sliding-window inference over the full volume and reports
    per-case Dice and HD95, averaged over cases (the BraTS convention);
  * scalar tasks (regression / classification) average predictions over a fixed,
    deterministic grid of patches instead of one random crop.

Nothing in this module samples randomness, so repeated evaluation of a checkpoint
returns identical numbers.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import TASK_CONFIGS


# --------------------------------------------------------------------------------------
# volume helpers
# --------------------------------------------------------------------------------------

def _load_volume(path: str) -> np.ndarray:
    vol = np.load(path + ".npy" if not path.endswith(".npy") else path, allow_pickle=True)
    if vol.ndim == 3:
        vol = vol[np.newaxis, ...]
    return np.asarray(vol, dtype=np.float32)


def _load_seg(path: str) -> Optional[np.ndarray]:
    import os
    seg_path = path + "_seg.npy"
    if not os.path.exists(seg_path):
        return None
    seg = np.load(seg_path, allow_pickle=True)
    if seg.ndim == 4:
        seg = seg[0]
    return (np.asarray(seg) > 0).astype(np.uint8)


def _load_scalar(path: str) -> float:
    return float(np.atleast_1d(np.loadtxt(path + ".txt", dtype=np.float64))[0])


def _patch_starts(size: int, patch: int, step: int) -> List[int]:
    """Start offsets covering ``size`` with ``patch``-wide windows, last one flush to the end."""
    if size <= patch:
        return [0]
    starts = list(range(0, size - patch + 1, step))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _gaussian_weight(patch: Tuple[int, int, int], sigma_scale: float = 0.125) -> np.ndarray:
    """Gaussian importance map so window seams do not create hard edges."""
    axes = []
    for p in patch:
        centre = (p - 1) / 2.0
        sigma = p * sigma_scale
        x = np.arange(p, dtype=np.float64)
        axes.append(np.exp(-((x - centre) ** 2) / (2 * sigma ** 2)))
    w = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return (w / w.max()).astype(np.float32)


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-case Dice. Two empty masks count as perfect agreement, the BraTS convention."""
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / denom)


def hd95(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float] = (1.0, 1.0, 1.0)) -> float:
    """95th-percentile symmetric Hausdorff distance in voxel units.

    Returns NaN when exactly one of the masks is empty (the distance is undefined and
    substituting a sentinel would silently distort the mean).
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float("nan")

    def surface(m):
        return m ^ binary_erosion(m, iterations=1, border_value=0)

    sp, sg = surface(p), surface(g)
    dt_to_g = distance_transform_edt(~g, sampling=spacing)
    dt_to_p = distance_transform_edt(~p, sampling=spacing)
    d_pg = dt_to_g[sp]
    d_gp = dt_to_p[sg]
    if d_pg.size == 0 or d_gp.size == 0:
        return float("nan")
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


# --------------------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------------------

@torch.no_grad()
def sliding_window_logits(
    model: torch.nn.Module,
    vol: np.ndarray,
    patch: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    batch_size: int = 8,
    device: torch.device = torch.device("cuda"),
    n_classes: int = 2,
    max_channels: Optional[int] = None,
) -> np.ndarray:
    """Gaussian-weighted sliding-window inference. Returns (n_classes, D, H, W) probabilities."""
    if max_channels is not None and vol.shape[0] > max_channels:
        vol = vol[:max_channels]

    c, *spatial = vol.shape
    pad = [max(0, patch[i] - spatial[i]) for i in range(3)]
    if any(pad):
        vol = np.pad(vol, ((0, 0), (0, pad[0]), (0, pad[1]), (0, pad[2])))
        spatial = list(vol.shape[1:])

    step = [max(1, int(patch[i] * (1 - overlap))) for i in range(3)]
    starts = [_patch_starts(spatial[i], patch[i], step[i]) for i in range(3)]
    coords = [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]

    acc = np.zeros((n_classes, *spatial), dtype=np.float32)
    norm = np.zeros(spatial, dtype=np.float32)
    weight = _gaussian_weight(patch)

    for i in range(0, len(coords), batch_size):
        chunk = coords[i : i + batch_size]
        batch = np.stack([
            vol[:, z : z + patch[0], y : y + patch[1], x : x + patch[2]] for z, y, x in chunk
        ])
        out = model(torch.from_numpy(batch).to(device))
        prob = F.softmax(out, dim=1).cpu().numpy()
        for j, (z, y, x) in enumerate(chunk):
            acc[:, z : z + patch[0], y : y + patch[1], x : x + patch[2]] += prob[j] * weight
            norm[z : z + patch[0], y : y + patch[1], x : x + patch[2]] += weight

    acc /= np.maximum(norm, 1e-8)
    if any(pad):
        acc = acc[:, : spatial[0] - pad[0], : spatial[1] - pad[1], : spatial[2] - pad[2]]
    return acc


@torch.no_grad()
def scalar_predict(
    model: torch.nn.Module,
    vol: np.ndarray,
    patch: Tuple[int, int, int] = (64, 64, 64),
    grid: Tuple[int, int, int] = (2, 2, 2),
    batch_size: int = 8,
    device: torch.device = torch.device("cuda"),
    max_channels: Optional[int] = None,
) -> np.ndarray:
    """Average the model output over a fixed grid of patches (deterministic, no cropping RNG)."""
    if max_channels is not None and vol.shape[0] > max_channels:
        vol = vol[:max_channels]

    c, *spatial = vol.shape
    pad = [max(0, patch[i] - spatial[i]) for i in range(3)]
    if any(pad):
        vol = np.pad(vol, ((0, 0), (0, pad[0]), (0, pad[1]), (0, pad[2])))
        spatial = list(vol.shape[1:])

    starts = []
    for i in range(3):
        n = max(1, grid[i])
        span = spatial[i] - patch[i]
        if span <= 0:
            starts.append([0])
        else:
            starts.append([int(round(span * k / max(1, n - 1))) if n > 1 else span // 2
                           for k in range(n)])
    coords = [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]

    outs = []
    for i in range(0, len(coords), batch_size):
        chunk = coords[i : i + batch_size]
        batch = np.stack([
            vol[:, z : z + patch[0], y : y + patch[1], x : x + patch[2]] for z, y, x in chunk
        ])
        outs.append(model(torch.from_numpy(batch).to(device)).float().cpu().numpy())
    return np.concatenate(outs, axis=0).mean(axis=0)


# --------------------------------------------------------------------------------------
# split-level evaluation
# --------------------------------------------------------------------------------------

def evaluate_split(
    model: torch.nn.Module,
    files: Sequence[str],
    task_id: int,
    device: torch.device,
    patch: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    batch_size: int = 8,
    max_channels: Optional[int] = None,
    return_per_case: bool = False,
) -> Dict[str, object]:
    """Evaluate a checkpoint on a list of cases and return aggregate (and optional per-case) metrics."""
    cfg = TASK_CONFIGS[task_id]
    task_type = cfg["task_type"]
    model.eval()

    if task_type == "segmentation":
        dices, hds, per_case = [], [], []
        for f in files:
            vol = _load_volume(f)
            gt = _load_seg(f)
            if gt is None:
                continue
            prob = sliding_window_logits(
                model, vol, patch=patch, overlap=overlap, batch_size=batch_size,
                device=device, n_classes=cfg["num_classes"], max_channels=max_channels,
            )
            pred = (prob.argmax(0) > 0).astype(np.uint8)
            gt = gt[: pred.shape[0], : pred.shape[1], : pred.shape[2]]
            d, h = dice_score(pred, gt), hd95(pred, gt)
            dices.append(d)
            hds.append(h)
            per_case.append({"case": f, "dice": d, "hd95": h})
        hd_arr = np.array(hds, dtype=float)
        out = {
            "n": len(dices),
            "dice": float(np.mean(dices)) if dices else float("nan"),
            "dice_std": float(np.std(dices)) if dices else float("nan"),
            "dice_median": float(np.median(dices)) if dices else float("nan"),
            "hd95": float(np.nanmean(hd_arr)) if hd_arr.size else float("nan"),
            "hd95_undefined": int(np.isnan(hd_arr).sum()),
        }
        if return_per_case:
            out["per_case"] = per_case
        return out

    preds, gts = [], []
    for f in files:
        vol = _load_volume(f)
        out = scalar_predict(model, vol, patch=patch, batch_size=batch_size,
                             device=device, max_channels=max_channels)
        preds.append(out)
        gts.append(_load_scalar(f))
    gts = np.asarray(gts, dtype=np.float64)

    if task_type == "regression":
        p = np.asarray([float(np.ravel(x)[0]) for x in preds], dtype=np.float64)
        resid = p - gts
        ss_res = float((resid ** 2).sum())
        ss_tot = float(((gts - gts.mean()) ** 2).sum())
        # Slope of predicted-on-true age: a constant predictor gives 0, an unbiased one 1.
        slope = float(np.polyfit(gts, p, 1)[0]) if gts.std() > 0 else float("nan")
        res = {
            "n": int(len(p)),
            "mae": float(np.abs(resid).mean()),
            "rmse": float(np.sqrt((resid ** 2).mean())),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "pearson_r": float(np.corrcoef(p, gts)[0, 1]) if p.std() > 0 and gts.std() > 0 else float("nan"),
            "bias": float(resid.mean()),
            "pred_std": float(p.std()),
            "gt_std": float(gts.std()),
            "calibration_slope": slope,
        }
        if return_per_case:
            res["per_case"] = [{"case": f, "pred": float(pi), "gt": float(gi)}
                               for f, pi, gi in zip(files, p, gts)]
        return res

    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    logits = np.stack([np.ravel(x) for x in preds])
    score = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
    y = gts.astype(int)
    res = {
        "n": int(len(y)),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        "balanced_acc": float(balanced_accuracy_score(y, (score > 0.5).astype(int))),
        "pos_rate": float(y.mean()),
    }
    if return_per_case:
        res["per_case"] = [{"case": f, "score": float(s), "gt": int(g)}
                           for f, s, g in zip(files, score, y)]
    return res
