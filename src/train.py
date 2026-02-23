"""
Training loop for single-task and continual learning.
Includes EWC, LwF, and Experience Replay for continual learning baselines.
"""
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple

from sklearn.metrics import roc_auc_score


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Count number of parameters (optionally trainable only)."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def train_task(
    model: nn.Module,
    train_loader,
    val_loader,
    task_id: int,
    task_type: str,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    save_dir: Optional[str] = None,
    log_interval: int = 10,
) -> Dict[str, float]:
    """Train model on a single task. Returns dict with best val metrics, time_sec, gpu_max_mb."""
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-6)

    if task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
    elif task_type == "segmentation":
        def _dice_ce(pred, target):
            target = target.squeeze(1).long().clamp(0, 1)
            ce = F.cross_entropy(pred, target, ignore_index=255)
            pred_soft = F.softmax(pred, dim=1)[:, 1]
            dice = 1 - (2 * (pred_soft * (target.float())).sum() + 1e-8) / (pred_soft.sum() + target.float().sum() + 1e-8)
            return ce + dice
        loss_fn = _dice_ce
    else:
        loss_fn = nn.MSELoss()

    best_val = {}
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"E{epoch+1}", leave=False):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            if task_type == "regression" and y.dim() == 1:
                y = y.unsqueeze(1)
            optim.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_metrics = evaluate(model, val_loader, task_id, task_type, device)
        val_metrics["loss"] = total_loss / max(n_batches, 1)

        key = "auc" if task_type == "classification" else ("dice" if task_type == "segmentation" else "mae")
        better = not best_val or (
            val_metrics[key] > best_val.get(key, -1) if key != "mae" else val_metrics[key] < best_val.get(key, float("inf"))
        )
        if better:
            best_val = val_metrics.copy()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save({"model_state": model.state_dict(), "task_id": task_id}, os.path.join(save_dir, f"task{task_id}_best.pt"))

        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1} | val={val_metrics}")
    elapsed = time.perf_counter() - t0
    best_val["time_sec"] = float(elapsed)
    best_val["n_params"] = count_params(model, trainable_only=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        best_val["gpu_max_mb"] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
    return best_val


def evaluate(model: nn.Module, val_loader, task_id: int, task_type: str, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["label"]
            out = model(x)
            if task_type == "classification":
                all_preds.append(out.softmax(1)[:, 1].cpu())
                all_labels.append(y)
            elif task_type == "regression":
                all_preds.append(out.cpu())
                y = y.float().unsqueeze(1) if y.dim() == 1 else y.float()
                all_labels.append(y)
            else:
                all_preds.append(out.argmax(1).float().cpu())
                all_labels.append(y.squeeze(1).float().cpu())
    if task_type == "classification":
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        auc = roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else 0.5
        return {"auc": auc}
    elif task_type == "regression":
        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        return {"mae": float(np.abs(preds - labels).mean())}
    else:
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        dice = (2 * (preds * labels).sum() + 1e-8) / (preds.sum() + labels.sum() + 1e-8)
        return {"dice": float(dice.item())}


# --- EWC ---
def compute_fisher_diagonal(
    model: nn.Module,
    train_loader,
    task_id: int,
    task_type: str,
    device: torch.device,
    n_samples: int = 200,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Compute diagonal Fisher information matrix (approximation) and optimal params.
    Returns (fisher, optimal_params) dicts keyed by param name.
    """
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    optimal = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    n_done = 0
    for batch in train_loader:
        if n_done >= n_samples:
            break
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        if task_type == "regression" and y.dim() == 1:
            y = y.unsqueeze(1)
        model.zero_grad()
        out = model(x)
        if task_type == "classification":
            log_prob = F.log_softmax(out, dim=1)
            for i in range(x.size(0)):
                if n_done >= n_samples:
                    break
                log_prob[i, y[i].item()].backward(retain_graph=(i < x.size(0) - 1))
                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.data.pow(2)
                n_done += 1
        elif task_type == "segmentation":
            target = y.squeeze(1).long().clamp(0, 1)
            for b in range(x.size(0)):
                if n_done >= n_samples:
                    break
                model.zero_grad()
                out_b = model(x[b : b + 1])
                log_prob = F.log_softmax(out_b, dim=1)
                ce = F.nll_loss(log_prob, target[b : b + 1], ignore_index=255)
                ce.backward()
                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.data.pow(2)
                n_done += 1
        else:  # regression
            loss = F.mse_loss(out, y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
            n_done += x.size(0)

    for n in fisher:
        fisher[n] /= max(n_done, 1)
    return fisher, optimal


def ewc_penalty(model: nn.Module, fisher: dict, optimal: dict) -> torch.Tensor:
    """Compute EWC regularization term."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if n in fisher and p.requires_grad:
            fi = fisher[n].to(p.device)
            opt = optimal[n].to(p.device)
            loss = loss + (fi * (p - opt).pow(2)).sum()
    return loss


# --- LwF (feature distillation) ---
def get_encoder_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Extract bottleneck encoder features (before task head)."""
    backbone = model.backbone
    if hasattr(backbone, "encoder"):
        enc = backbone.encoder(x)
    else:
        enc = backbone(x)
    if isinstance(enc, (list, tuple)):
        return enc[-1]
    return enc


# --- Extended train_task with EWC, LwF, Replay ---
def train_task_cl(
    model: nn.Module,
    train_loader,
    val_loader,
    task_id: int,
    task_type: str,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    save_dir: Optional[str] = None,
    log_interval: int = 10,
    # EWC
    ewc_fisher: Optional[dict] = None,
    ewc_optimal: Optional[dict] = None,
    lambda_ewc: float = 1000.0,
    # LwF
    old_model: Optional[nn.Module] = None,
    distill_weight: float = 0.5,
    distill_temp: float = 2.0,
    # Replay
    replay_buffer=None,
    replay_task_ids: Optional[List[int]] = None,
    replay_weight: float = 0.5,
    replay_batch_size: int = 2,
) -> Dict[str, float]:
    """Train with optional EWC, LwF, or Replay."""
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optim, T_max=epochs, eta_min=1e-6)

    if task_type == "classification":
        loss_fn = nn.CrossEntropyLoss()
    elif task_type == "segmentation":
        def _dice_ce(pred, target):
            target = target.squeeze(1).long().clamp(0, 1)
            ce = F.cross_entropy(pred, target, ignore_index=255)
            pred_soft = F.softmax(pred, dim=1)[:, 1]
            dice = 1 - (2 * (pred_soft * (target.float())).sum() + 1e-8) / (pred_soft.sum() + target.float().sum() + 1e-8)
            return ce + dice
        loss_fn = _dice_ce
    else:
        loss_fn = nn.MSELoss()

    input_channels = next(iter(train_loader)).get("image").size(1)
    replay_iter = None
    if replay_buffer and replay_task_ids:
        replay_iter = replay_buffer.get_replay_loader(replay_task_ids, replay_batch_size, input_channels, device)

    best_val = {}
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"E{epoch+1}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            if task_type == "regression" and y.dim() == 1:
                y = y.unsqueeze(1)
            optim.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)

            # LwF: distill encoder features
            if old_model is not None:
                with torch.no_grad():
                    old_feat = get_encoder_features(old_model, x)
                new_feat = get_encoder_features(model, x)
                loss_distill = F.mse_loss(new_feat, old_feat.to(new_feat.device))
                loss = loss + distill_weight * loss_distill

            # Replay: LwF-style feature distillation on stored samples (avoids need for old task heads)
            if replay_iter is not None and old_model is not None:
                try:
                    x_r, y_r = next(replay_iter)
                except StopIteration:
                    replay_iter = replay_buffer.get_replay_loader(replay_task_ids, replay_batch_size, input_channels, device)
                    x_r, y_r = next(replay_iter)
                with torch.no_grad():
                    old_feat_r = get_encoder_features(old_model, x_r)
                new_feat_r = get_encoder_features(model, x_r)
                loss_replay = F.mse_loss(new_feat_r, old_feat_r.to(new_feat_r.device))
                loss = loss + replay_weight * loss_replay

            # EWC
            if ewc_fisher is not None and ewc_optimal is not None:
                loss = loss + lambda_ewc * ewc_penalty(model, ewc_fisher, ewc_optimal)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        val_metrics = evaluate(model, val_loader, task_id, task_type, device)
        val_metrics["loss"] = total_loss / max(n_batches, 1)
        key = "auc" if task_type == "classification" else ("dice" if task_type == "segmentation" else "mae")
        better = not best_val or (val_metrics[key] > best_val.get(key, -1) if key != "mae" else val_metrics[key] < best_val.get(key, float("inf")))
        if better:
            best_val = val_metrics.copy()
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save({"model_state": model.state_dict(), "task_id": task_id}, os.path.join(save_dir, f"task{task_id}_best.pt"))
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1} | val={val_metrics}")

    elapsed = time.perf_counter() - t0
    best_val["time_sec"] = float(elapsed)
    best_val["n_params"] = count_params(model, trainable_only=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        best_val["gpu_max_mb"] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
    return best_val
