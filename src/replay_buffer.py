"""
Replay buffer for experience replay in continual learning.
Stores (image, label, task_id) from previous tasks.
"""
import numpy as np
import torch
from collections import deque
from typing import List, Optional, Tuple
import random


class ReplayBuffer:
    """
    Buffer of samples from previous tasks for experience replay.
    When sampling for a task with different input channels, we slice/pad to match.
    """
    def __init__(self, capacity: int = 500, seed: int = 42):
        self.capacity = capacity
        self.buffers = {}  # task_id -> deque of (x, y) tensors
        self.rng = np.random.default_rng(seed)

    def add(self, task_id: int, images: torch.Tensor, labels: torch.Tensor):
        """Add a batch (images, labels) from task_id to the buffer."""
        if task_id not in self.buffers:
            self.buffers[task_id] = deque(maxlen=self.capacity)
        for i in range(images.size(0)):
            if len(self.buffers[task_id]) >= self.capacity:
                break
            x = images[i : i + 1].cpu().clone()
            if labels.dim() > 1:
                y = labels[i : i + 1].cpu().clone()
            else:
                y = torch.tensor([labels[i].item()], dtype=labels.dtype).unsqueeze(1)
            self.buffers[task_id].append((x, y))

    def add_from_loader(self, task_id: int, dataloader, max_samples: Optional[int] = None):
        """Populate buffer from a dataloader (e.g., after training a task)."""
        n = 0
        for batch in dataloader:
            x = batch["image"]
            y = batch["label"]
            if y.dim() == 1:
                y = y.unsqueeze(1)
            self.add(task_id, x, y)
            n += x.size(0)
            if max_samples and n >= max_samples:
                break

    def sample(self, task_id: int, batch_size: int, target_channels: Optional[int] = None) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample a batch from the buffer for given task_id.
        If target_channels is set and differs from stored channels, we slice or pad the channel dim.
        """
        if task_id not in self.buffers or len(self.buffers[task_id]) == 0:
            return None
        buf = list(self.buffers[task_id])
        idxs = self.rng.choice(len(buf), size=min(batch_size, len(buf)), replace=False)
        xs, ys = [], []
        for i in idxs:
            x, y = buf[i]
            xs.append(x)
            ys.append(y)
        x_batch = torch.cat(xs, dim=0)
        y_batch = torch.cat(ys, dim=0) if ys[0].dim() > 1 else torch.stack([y.squeeze() for y in ys]).unsqueeze(1)
        if target_channels is not None and x_batch.size(1) != target_channels:
            c_in, c_tgt = x_batch.size(1), target_channels
            if c_in >= c_tgt:
                x_batch = x_batch[:, :c_tgt]
            else:
                pad = torch.zeros(x_batch.size(0), c_tgt - c_in, *x_batch.shape[2:], device=x_batch.device, dtype=x_batch.dtype)
                x_batch = torch.cat([x_batch, pad], dim=1)
        return x_batch, y_batch

    def get_replay_loader(self, task_ids: List[int], batch_size: int, target_channels: int, device, infinite: bool = True):
        """
        Yields batches of replay data for the given task_ids.
        Used during training to mix with current task.
        """
        if not task_ids:
            return
        combined = []
        for tid in task_ids:
            if tid in self.buffers:
                combined.extend([(tid, t) for t in self.buffers[tid]])
        if not combined:
            return
        while True:
            random.shuffle(combined)
            for i in range(0, len(combined), batch_size):
                batch = combined[i : i + batch_size]
                xs, ys = [], []
                for tid, (x, y) in batch:
                    xs.append(x)
                    ys.append(y)
                x_batch = torch.cat(xs, dim=0)
                y_batch = torch.cat(ys, dim=0) if ys[0].dim() > 1 else torch.stack([y.squeeze() for y in ys])
                if x_batch.size(1) != target_channels:
                    c_in, c_tgt = x_batch.size(1), target_channels
                    if c_in >= c_tgt:
                        x_batch = x_batch[:, :c_tgt]
                    else:
                        pad = torch.zeros(x_batch.size(0), c_tgt - c_in, *x_batch.shape[2:], dtype=x_batch.dtype)
                        x_batch = torch.cat([x_batch, pad], dim=1)
                yield x_batch.to(device), y_batch.to(device)
            if not infinite:
                break

    def __len__(self):
        return sum(len(b) for b in self.buffers.values())
