"""
Backbone wrapper: load FOMO pretrained UNet, freeze, optionally inject LoRA.
"""
import sys
from pathlib import Path

# Ensure fomo_baseline is importable
_root = Path(__file__).resolve().parent.parent
_fomo = _root / "fomo_baseline"
if str(_fomo) not in sys.path:
    sys.path.insert(0, str(_fomo))

import torch
import torch.nn as nn
from typing import Optional, Literal

from models.networks.unet import unet_b, unet_xl
from utils.utils import load_pretrained_weights


def build_backbone(
    model_name: str = "unet_b",
    mode: Literal["enc", "segmentation", "classification", "regression"] = "enc",
    input_channels: int = 4,  # Task1: 4, Task2: 3, Task3: 2
    output_channels: int = 1,
    pretrained_path: Optional[str] = None,
    freeze: bool = True,
    inject_lora: bool = False,
    lora_r: int = 8,
) -> nn.Module:
    """
    Build backbone (encoder or full UNet).
    mode='enc': encoder only, returns list of features [x0,...,x4].
    mode='segmentation'/'classification'/'regression': full model with head.
    """
    if model_name == "unet_b":
        if mode == "enc":
            net = unet_b(mode="enc", input_channels=input_channels, output_channels=output_channels)
        else:
            net = unet_b(
                mode=mode,
                input_channels=input_channels,
                output_channels=output_channels,
            )
    elif model_name == "unet_xl":
        if mode == "enc":
            net = unet_xl(mode="enc", input_channels=input_channels, output_channels=output_channels)
        else:
            net = unet_xl(
                mode=mode,
                input_channels=input_channels,
                output_channels=output_channels,
            )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if pretrained_path:
        state_dict = load_pretrained_weights(pretrained_path, compile_flag=False)
        # Filter keys that match (handle output head mismatch)
        model_dict = net.state_dict()
        load_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        net.load_state_dict(load_dict, strict=False)
        print(f"Loaded {len(load_dict)}/{len(model_dict)} keys from {pretrained_path}")

    if freeze:
        for p in net.parameters():
            p.requires_grad = False

    if inject_lora:
        try:
            from src.lora import inject_lora_into_unet
        except ImportError:
            from lora import inject_lora_into_unet
        net = inject_lora_into_unet(net, r=lora_r)

    return net
