#!/usr/bin/env python3
"""
Measure peak VRAM for the heaviest step of a benchmark run, so the gq declaration is
based on an observation rather than a guess.

Covers both phases: a training step at the real batch size, and one whole-volume
sliding-window evaluation at the real eval batch size (the part that was never measured).

  gq run -x -- python scripts/probe_vram.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))

from src.data import TASK_CONFIGS, FewShotFOMODataset, _seg_collate  # noqa: E402
from src.eval import evaluate_split  # noqa: E402
from src.models import TaskModel  # noqa: E402
from src.splits import load_or_make_splits  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def mb(x):
    return x / 1024 / 1024


def report(tag):
    torch.cuda.synchronize()
    alloc = mb(torch.cuda.max_memory_allocated())
    resv = mb(torch.cuda.max_memory_reserved())
    print(f"  {tag:<34} allocated {alloc:8.0f} MB   reserved {resv:8.0f} MB", flush=True)
    return resv


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    device = torch.device("cuda")
    patch = (64, 64, 64)
    peak = 0.0

    for task_id, train_bs, eval_bs in [(2, 2, 8), (3, 2, 8)]:
        cfg = TASK_CONFIGS[task_id]
        print(f"\n=== task {task_id} ({cfg['task_type']}) ===", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = TaskModel(task_id, cfg["modalities"], patch,
                          pretrained_path=str(ROOT / "weights/fomo25_mmunetvae_pretrained.ckpt"),
                          model_name="unet_b", lora_r=8, lora_decoder=True).to(device)

        splits = load_or_make_splits(str(ROOT / "data/preprocessed"), task_id, 32, seed=42,
                                     split_dir=str(ROOT / "data/splits_v2"))

        # --- training step at the real batch size
        collate = _seg_collate if cfg["task_type"] == "segmentation" else None
        ds = FewShotFOMODataset(splits["train"], patch, cfg["task_type"])
        loader = DataLoader(ds, batch_size=train_bs, shuffle=True, collate_fn=collate)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        model.train()
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            out = model(x)
            if cfg["task_type"] == "segmentation":
                loss = torch.nn.functional.cross_entropy(out, y.squeeze(1).long().clamp(0, 1))
            else:
                loss = torch.nn.functional.mse_loss(out, y.float().view(-1, 1))
            loss.backward()
            opt.step()
            opt.zero_grad()
            break
        peak = max(peak, report(f"train step (batch {train_bs})"))

        # --- one whole-volume evaluation at the real eval batch size
        t0 = time.perf_counter()
        evaluate_split(model, splits["test"][:1], task_id, device, patch=patch,
                       overlap=0.5, batch_size=eval_bs, max_channels=cfg["modalities"])
        secs = time.perf_counter() - t0
        peak = max(peak, report(f"sliding-window eval (batch {eval_bs})"))
        print(f"  one case took {secs:.1f}s -> ~{secs * 250 / 60:.0f} min for a 250-case split",
              flush=True)

        del model, opt, loader, ds
        torch.cuda.empty_cache()

    print(f"\nPEAK RESERVED ACROSS BOTH TASKS: {peak:.0f} MB")
    print(f"suggested gq declaration (+20%): -g {int(np.ceil(peak * 1.2 / 1024))}G")


if __name__ == "__main__":
    main()
