"""
Champion (jbanusco) mmunetvae backbone + LoRA for continual learning.
Requires: champion_fomo repo, yucca (or yucca_stub in path).
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_champion = _root / "champion_fomo" / "src"
_yucca_stub = _root / "yucca_stub"

# Prepend yucca stub before any other imports (for get_steps_for_sliding_window)
if _yucca_stub.exists() and str(_yucca_stub) not in sys.path:
    sys.path.insert(0, str(_yucca_stub))
if str(_champion) not in sys.path:
    sys.path.insert(0, str(_champion))

import torch
import torch.nn as nn
from typing import Optional

from models.networks.mmunetvae import mmunetvae


def _inject_lora(module, r, lora_alpha, lora_dropout):
    try:
        from src.lora import _replace_conv3d_with_lora
    except ImportError:
        from lora import _replace_conv3d_with_lora
    _replace_conv3d_with_lora(module, "", r, lora_alpha, lora_dropout)
    for n, p in module.named_parameters():
        if "lora_" in n:
            p.requires_grad = True


class ChampionTaskModel(nn.Module):
    """
    FOMO25 champion mmunetvae + LoRA for continual learning.
    Forward: x [B,C,D,H,W] -> logits/pred (task_output).
    """
    def __init__(
        self,
        task_id: int,
        input_channels: int,
        patch_size: tuple,
        pretrained_path: Optional[str] = None,
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        use_skip_connections: bool = False,
    ):
        super().__init__()
        self.task_id = task_id
        self.task_type = {1: "classification", 2: "segmentation", 3: "regression"}[task_id]
        output_channels = 2 if task_id == 2 else (2 if task_id == 1 else 1)

        self.model = mmunetvae(
            input_channels=input_channels,
            output_channels=output_channels,
            mode=self.task_type,
            use_vae=False,
            use_skip_connections=use_skip_connections and (task_id == 2),
        )

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        for p in self.model.parameters():
            p.requires_grad = False

        if lora_r and lora_r > 0 and hasattr(self.model, "encoder"):
            _inject_lora(self.model.encoder, lora_r, lora_alpha, lora_dropout)
            if hasattr(self.model, "decoder_task") and self.model.decoder_task is not None:
                if hasattr(self.model.decoder_task, "encoder") or "decoder" in str(type(self.model.decoder_task)):
                    _inject_lora(self.model.decoder_task, lora_r, lora_alpha, lora_dropout)

        for n, p in self.model.named_parameters():
            if "decoder_task" in n or "fusion" in n or "lora_" in n:
                p.requires_grad = True

    def _load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        if isinstance(sd, dict) and any("_orig_mod" in k for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model_sd = self.model.state_dict()
        load_sd = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        self.model.load_state_dict(load_sd, strict=False)
        print(f"Champion: loaded {len(load_sd)}/{len(model_sd)} keys from {path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x_list=x)
        return out["task_output"]
