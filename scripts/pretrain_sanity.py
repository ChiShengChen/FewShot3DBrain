#!/usr/bin/env python3
"""
Does the FOMO checkpoint actually load, and does it change anything? (reviewer control)

The pretrained and randomly initialised backbones give identical LoRA + head results on BraTS
(0.772 vs 0.772 Dice) and IXI brain age (10.99 vs 10.99 yr). Before reading that as "pretraining
does not help", rule out a silent loading failure. This script reports, for the segmentation
U-Net (task 2) and the encoder-only model (task 5):

  (a) which checkpoint tensors load into the model and which model tensors stay at init;
  (b) how far each loaded tensor moved from a fresh random initialisation (relative L2);
  (c) pooled deepest-stage feature statistics on the 32 seed-42 training subjects of task 5,
      pretrained vs random: mean pairwise cosine and between-subject SD as a fraction of the
      mean absolute feature;
  (d) the untrained model's starting point on the seed-42 validation split: BraTS Dice of the
      untrained U-Net (pretrained decoder vs random), pretrained vs random.

  python scripts/pretrain_sanity.py      # -> outputs/benchmark_v2/sanity/pretrain_sanity.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fomo_baseline" / "src"))

from src.data import TASK_CONFIGS  # noqa: E402
from src.eval import _load_volume, evaluate_split  # noqa: E402
from src.models import TaskModel, _load_pretrained_weights  # noqa: E402
from src.splits import load_or_make_splits  # noqa: E402

CKPT = str(ROOT / "weights/fomo25_mmunetvae_pretrained.ckpt")
OUT = ROOT / "outputs/benchmark_v2/sanity"


def build(task, pretrained, seed=0, patch=64):
    torch.manual_seed(seed)
    cfg = TASK_CONFIGS[task]
    return TaskModel(task, cfg["modalities"], (patch,) * 3, pretrained_path=CKPT if pretrained else None,
                     model_name="unet_b", lora_r=0, lora_decoder=False)


def load_report(task):
    sd = _load_pretrained_weights(CKPT)
    if any("_orig_mod" in k for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    if any(k.startswith("model.") for k in sd):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
    rnd = build(task, False)
    pre = build(task, True)
    msd = rnd.backbone.state_dict()
    loaded = [k for k, v in sd.items() if k in msd and msd[k].shape == v.shape]
    shape_mismatch = [k for k, v in sd.items() if k in msd and msd[k].shape != v.shape]
    not_in_ckpt = [k for k in msd if k not in sd]
    rel = {}
    psd = pre.backbone.state_dict()
    for k in loaded:
        a, b = psd[k].float(), msd[k].float()
        if b.numel() > 1 and b.norm() > 0:
            rel[k] = float((a - b).norm() / b.norm())
    # Tensors whose random init is constant (norm affine weight = 1, bias = 0) are compared by
    # their distance from that constant instead.
    const_init = {k: float((psd[k].float() - msd[k].float()).abs().mean()) for k in loaded if k not in rel}
    return {
        "model_tensors": len(msd), "loaded": len(loaded), "shape_mismatch": shape_mismatch,
        "not_in_checkpoint": not_in_ckpt,
        "rel_change_vs_random": {"min": min(rel.values()), "median": float(np.median(list(rel.values()))),
                                 "max": max(rel.values()), "n": len(rel)},
        "const_init_tensors_mean_abs_change": {"n": len(const_init),
                                               "median": float(np.median(list(const_init.values()))) if const_init else None},
    }


@torch.no_grad()
def feature_stats(task, pretrained, device, patch=96, seed=42):
    model = build(task, pretrained, patch=patch).to(device).eval()
    sp = load_or_make_splits(str(ROOT / "data/preprocessed"), task, 32, seed=seed, split_dir=str(ROOT / "data/splits_v3"))
    F = []
    for f in sp["train"]:
        vol = _load_volume(f)[: TASK_CONFIGS[task]["modalities"]]
        enc = model.backbone(torch.from_numpy(np.ascontiguousarray(vol))[None].to(device))
        F.append(enc[-1].mean(dim=(2, 3, 4)).squeeze(0).float().cpu().numpy())
    F = np.stack(F)
    Fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    cos = Fn @ Fn.T
    iu = np.triu_indices(len(F), 1)
    return {"n_subjects": len(F), "mean_pairwise_cosine": float(cos[iu].mean()),
            "between_subject_sd_frac": float(F.std(0).mean() / np.abs(F).mean()),
            "mean_abs_feature": float(np.abs(F).mean())}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"checkpoint": CKPT}
    for task in (2, 5):
        res[f"load_task{task}"] = load_report(task)
    res["features_task5"] = {"pretrained": feature_stats(5, True, device), "random": feature_stats(5, False, device)}
    sp = load_or_make_splits(str(ROOT / "data/preprocessed"), 2, 32, seed=42, split_dir=str(ROOT / "data/splits_v3"))
    start = {}
    for name, pre in (("pretrained", True), ("random", False)):
        m = build(2, pre).to(device)
        r = evaluate_split(m, sp["val"], 2, device, patch=(64,) * 3, overlap=0.25, batch_size=8, max_channels=3)
        start[name] = {"val_dice": r["dice"], "n": r["n"]}
    res["untrained_brats_val"] = start
    (OUT / "pretrain_sanity.json").write_text(json.dumps(res, indent=2))

    print("## Pretrained checkpoint sanity check\n")
    for task in (2, 5):
        L = res[f"load_task{task}"]
        print(f"- task {task}: {L['loaded']}/{L['model_tensors']} model tensors loaded; "
              f"not in checkpoint: {len(L['not_in_checkpoint'])} {L['not_in_checkpoint'][:6]}; "
              f"shape mismatch: {L['shape_mismatch']}")
        r = L["rel_change_vs_random"]
        print(f"  relative L2 distance pretrained vs random init over {r['n']} tensors: "
              f"min {r['min']:.2f}, median {r['median']:.2f}, max {r['max']:.2f}")
    for k, v in res["features_task5"].items():
        print(f"- IXI pooled features ({k}): mean pairwise cosine {v['mean_pairwise_cosine']:.5f}, "
              f"between-subject SD {100 * v['between_subject_sd_frac']:.2f}% of mean |feature|")
    for k, v in start.items():
        print(f"- untrained U-Net ({k}) BraTS val Dice {v['val_dice']:.3f} (n={v['n']})")
    print(f"\nwrote {OUT / 'pretrain_sanity.json'}")


if __name__ == "__main__":
    main()
