"""
Houlsby-style bottleneck adapters for 3D convolutions.

A second parameter-efficient comparator for LoRA at a matched budget. Where LoRA adds a
low-rank *parallel* update to a frozen convolution, an adapter adds a small *serial*
bottleneck after it: y = conv(x) + up(act(down(y))), with `down` a 1x1x1 projection to
`dim` channels, `up` a 1x1x1 projection back, and `up` zero-initialised so training
starts from the frozen network's function (Houlsby et al., 2019). Inserted after every
3x3x3 Conv3d of the encoder and decoder, exactly where LoRA attaches, and never on the
1x1x1 output projection, which every strategy trains on its own.

Per layer the adapter costs 2*C*dim + C + dim parameters; at dim=16 that is 93.7K on the
segmentation backbone against LoRA r=16's 93.8K, so the two are directly comparable.
"""
from typing import Optional

import torch
import torch.nn as nn


class Conv3dAdapter(nn.Module):
    def __init__(self, conv: nn.Conv3d, dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.conv = conv
        c = conv.out_channels
        self.adapter_down = nn.Conv3d(c, dim, kernel_size=1, bias=True)
        self.adapter_up = nn.Conv3d(dim, c, kernel_size=1, bias=True)
        self.adapter_act = nn.GELU()
        self.adapter_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.adapter_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)       # identity at initialisation
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y + self.adapter_up(self.adapter_drop(self.adapter_act(self.adapter_down(y))))


def _replace_conv3d_with_adapter(module: nn.Module, name: str, dim: int, dropout: float) -> int:
    """Recursively wrap every non-1x1x1 Conv3d with a bottleneck adapter. Returns count."""
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{name}.{child_name}" if name else child_name
        if isinstance(child, nn.Conv3d):
            if child.kernel_size != (1, 1, 1):
                setattr(module, child_name, Conv3dAdapter(child, dim=dim, dropout=dropout))
                replaced += 1
        elif isinstance(child, Conv3dAdapter):
            continue
        else:
            replaced += _replace_conv3d_with_adapter(child, full_name, dim, dropout)
    return replaced


def inject_adapters(module: nn.Module, dim: int = 16, dropout: float = 0.1) -> int:
    n = _replace_conv3d_with_adapter(module, "", dim, dropout)
    for pname, p in module.named_parameters():
        if "adapter_" in pname:
            p.requires_grad = True
    return n
