"""
Task-specific models: frozen backbone + LoRA + task head.
Supports classification (T1), segmentation (T2), regression (T3).
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_fomo = _root / "fomo_baseline"
if str(_fomo) not in sys.path:
    sys.path.insert(0, str(_fomo))

import torch
import torch.nn as nn
from typing import Optional, Literal

from models.networks.unet import unet_b, unet_xl
from models.networks.heads import ClsRegHead

def _load_pretrained_weights(path: str):
    """Load state dict from checkpoint (PyTorch Lightning or raw)."""
    try:
        from utils.utils import load_pretrained_weights
        return load_pretrained_weights(path, compile_flag=False)
    except ImportError:
        ckpt = torch.load(path, map_location="cpu")
        return ckpt.get("state_dict", ckpt)


def _inject_lora(module, r, lora_alpha, lora_dropout):
    """Inject LoRA into Conv3d layers."""
    try:
        from src.lora import _replace_conv3d_with_lora
    except ImportError:
        from lora import _replace_conv3d_with_lora
    _replace_conv3d_with_lora(module, "", r, lora_alpha, lora_dropout)
    for n, p in module.named_parameters():
        if "lora_" in n:
            p.requires_grad = True


class TaskModel(nn.Module):
    """
    Frozen UNet backbone + task-specific LoRA + task head.
    """
    def __init__(
        self,
        task_id: int,
        input_channels: int,
        patch_size: tuple,
        pretrained_path: Optional[str] = None,
        model_name: str = "unet_b",
        lora_r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.1,
        lora_decoder: bool = True,
    ):
        super().__init__()
        self.task_id = task_id
        self.task_type = self._get_task_type(task_id)
        
        if task_id == 1:
            # Classification: encoder + ClsRegHead
            self.backbone = self._build_encoder_only(model_name, input_channels, patch_size)
            self.head = ClsRegHead(in_channels=32 * 16, num_classes=2)  # unet_b starting_filters=32
        elif task_id == 2:
            # Segmentation: full UNet with decoder
            self.backbone = self._build_full_unet(model_name, input_channels, 2, patch_size)
            self.head = None
        elif task_id == 3:
            # Regression: encoder + regressor head
            self.backbone = self._build_encoder_only(model_name, input_channels, patch_size)
            self.head = ClsRegHead(in_channels=32 * 16, num_classes=1)
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Inject LoRA (skip if lora_r <= 0)
        if lora_r and lora_r > 0:
            if hasattr(self.backbone, "encoder"):
                _inject_lora(self.backbone.encoder, lora_r, lora_alpha, lora_dropout)
            if lora_decoder and hasattr(self.backbone, "decoder") and self.backbone.decoder is not None and not isinstance(self.backbone.decoder, nn.Identity):
                _inject_lora(self.backbone.decoder, lora_r, lora_alpha, lora_dropout)

        # Head is always trainable
        if self.head is not None:
            for p in self.head.parameters():
                p.requires_grad = True

    def _get_task_type(self, task_id: int) -> str:
        return {1: "classification", 2: "segmentation", 3: "regression"}[task_id]

    def _build_encoder_only(self, model_name: str, in_ch: int, patch_size: tuple) -> nn.Module:
        if model_name == "unet_b":
            from models.networks.unet import unet_b
            net = unet_b(mode="enc", input_channels=in_ch, output_channels=1)
        else:
            from models.networks.unet import unet_xl
            net = unet_xl(mode="enc", input_channels=in_ch, output_channels=1)
        return net

    def _build_full_unet(self, model_name: str, in_ch: int, out_ch: int, patch_size: tuple) -> nn.Module:
        if model_name == "unet_b":
            from models.networks.unet import unet_b
            net = unet_b(mode="segmentation", input_channels=in_ch, output_channels=out_ch)
        else:
            from models.networks.unet import unet_xl
            net = unet_xl(mode="segmentation", input_channels=in_ch, output_channels=out_ch)
        return net

    def _load_pretrained(self, path: str):
        sd = _load_pretrained_weights(path)
        if not isinstance(sd, dict):
            return
        # Handle PyTorch Lightning / compiled prefixes
        if any("_orig_mod" in k for k in sd.keys()):
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        # FOMO checkpoint uses "model.encoder." / "model.decoder." prefix
        if any(k.startswith("model.") for k in sd.keys()):
            sd = {k.replace("model.", "", 1): v for k, v in sd.items()}
        model_sd = self.backbone.state_dict()
        load_sd = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        self.backbone.load_state_dict(load_sd, strict=False)
        print(f"Loaded {len(load_sd)}/{len(model_sd)} pretrained keys")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task_id == 2:
            return self.backbone(x)  # full UNet for segmentation
        enc = self.backbone(x)  # list [x0..x4] for encoder mode
        return self.head(enc)
