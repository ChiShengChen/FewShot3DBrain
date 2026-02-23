"""
Custom LoRA for 3D Conv layers (PEFT targets Linear/Conv2d; we need Conv3d).
LoRA: W' = W + B @ A, where A in R^{r x k}, B in R^{d x r}.
For Conv3d: apply as two 1x1x1 convolutions (channel-wise low-rank).
"""
import math
import torch
import torch.nn as nn
from typing import Optional, List
import re


class Conv3dLoRA(nn.Module):
    """
    LoRA for 3D Convolution.
    Original: Conv3d(in_c, out_c, k) 
    LoRA: Conv1x1x1(in_c, r) -> Conv1x1x1(r, out_c), scaling by alpha/r.
    """
    def __init__(
        self,
        base_conv: nn.Conv3d,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.base_conv = base_conv
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        in_c = base_conv.in_channels
        out_c = base_conv.out_channels
        
        self.lora_A = nn.Conv3d(in_c, r, kernel_size=1, stride=1, padding=0, bias=False)
        self.lora_B = nn.Conv3d(r, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.lora_dropout = nn.Dropout3d(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_conv(x)
        lora_out = self.lora_dropout(x)
        lora_out = self.lora_A(lora_out)
        lora_out = self.lora_B(lora_out)
        return base_out + self.scaling * lora_out


def _replace_conv3d_with_lora(module: nn.Module, name: str, r: int, lora_alpha: float, lora_dropout: float) -> int:
    """Recursively replace Conv3d with Conv3dLoRA. Returns count replaced."""
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{name}.{child_name}" if name else child_name
        if isinstance(child, nn.Conv3d):
            # Skip 1x1 convolutions (e.g. out_conv) - focus on feature convs
            if child.kernel_size != (1, 1, 1):
                lora_conv = Conv3dLoRA(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                setattr(module, child_name, lora_conv)
                replaced += 1
        else:
            replaced += _replace_conv3d_with_lora(child, full_name, r, lora_alpha, lora_dropout)
    return replaced


def inject_lora_into_unet(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.1,
    target_submodules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Replace Conv3d layers in encoder/decoder with Conv3dLoRA.
    target_submodules: e.g. ["encoder", "decoder"]. If None, use both.
    """
    if target_submodules is None:
        target_submodules = ["encoder", "decoder"]
    
    replaced = 0
    for sub in target_submodules:
        if hasattr(model, sub):
            m = getattr(model, sub)
            replaced += _replace_conv3d_with_lora(m, sub, r, lora_alpha, lora_dropout)
    
    # Ensure LoRA params are trainable, base is frozen
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
        elif "base_conv" in n:
            p.requires_grad = False
    
    print(f"Injected LoRA into {replaced} Conv3d layers (r={r})")
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA parameters for saving."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def load_lora_state_dict(model: nn.Module, state_dict: dict, adapter_name: str = "default"):
    """Load LoRA weights. Keys may have adapter prefix."""
    model_dict = model.state_dict()
    loaded = 0
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            loaded += 1
    model.load_state_dict(model_dict, strict=False)
    return loaded
