"""
Second backbone for the backbone-independence check: MONAI Swin UNETR with the
self-supervised encoder weights of Tang et al. (CVPR 2022; CT, 5,050 scans).

Exposes the same interface as src.models.TaskModel so scripts/run_benchmark_v2.py can
swap it in with --backbone swinunetr, and the same *parameter names* the method
dispatch relies on:
  backbone.encoder.*   the Swin transformer (swinViT): frozen by every frozen strategy
  backbone.decoder.*   the convolutional skip blocks and up-blocks (the large task head)
  backbone.out_conv.*  the 1x1x1 output projection (98 params), trained in every strategy
  head.*               pooled regressor / classifier on the deepest encoder feature map
LoRA goes on every nn.Linear of the encoder (qkv, proj, MLP) and, when lora_decoder is
set, on every 3x3x3 Conv3d of the decoder, mirroring where it sits in the U-Net.
"""
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_root = Path(__file__).resolve().parent.parent
for _p in (_root / "fomo_baseline", _root / "fomo_baseline" / "src"):   # FOMO's `models` package
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from models.networks.heads import ClsRegHead  # noqa: E402  (FOMO's pooled head)

SWIN_FEATURE_SIZE = 48      # the self-supervised checkpoint was trained at this width
SWIN_DEEPEST = 16 * SWIN_FEATURE_SIZE   # 768 channels at the bottleneck


class SwinBackbone(nn.Module):
    """Swin UNETR re-grouped into encoder / decoder / out_conv (see module docstring)."""

    def __init__(self, in_channels: int, out_channels: int, mode: str):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        swin = SwinUNETR(in_channels=in_channels, out_channels=out_channels,
                         feature_size=SWIN_FEATURE_SIZE, use_v2=False)
        self.encoder = swin.swinViT
        self.normalize = swin.normalize
        self.mode = mode
        if mode == "segmentation":
            self.decoder = nn.ModuleDict({
                "skip0": swin.encoder1, "skip1": swin.encoder2, "skip2": swin.encoder3,
                "skip3": swin.encoder4, "bottleneck": swin.encoder10,
                "up5": swin.decoder5, "up4": swin.decoder4, "up3": swin.decoder3,
                "up2": swin.decoder2, "up1": swin.decoder1})
            self.out_conv = swin.out
        else:
            self.decoder = nn.Identity()
            self.out_conv = None

    def forward(self, x):
        hs = self.encoder(x, self.normalize)
        if self.mode != "segmentation":
            return hs                      # list of feature maps, deepest last (ClsRegHead uses x[-1])
        d = self.decoder                   # identical to monai SwinUNETR.forward
        enc0 = d["skip0"](x)
        enc1 = d["skip1"](hs[0])
        enc2 = d["skip2"](hs[1])
        enc3 = d["skip3"](hs[2])
        dec4 = d["bottleneck"](hs[4])
        dec3 = d["up5"](dec4, hs[3])
        dec2 = d["up4"](dec3, enc3)
        dec1 = d["up3"](dec2, enc2)
        dec0 = d["up2"](dec1, enc1)
        out = d["up1"](dec0, enc0)
        return self.out_conv(out)


class SwinTaskModel(nn.Module):
    """Drop-in for src.models.TaskModel on the Swin UNETR backbone."""

    def __init__(self, task_id: int, input_channels: int, patch_size: tuple,
                 pretrained_path: Optional[str] = None, model_name: str = "swinunetr",
                 lora_r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.1,
                 lora_decoder: bool = True, adapter_dim: int = 0, adapter_dropout: float = 0.1):
        super().__init__()
        if adapter_dim:
            raise NotImplementedError("bottleneck adapters are only run on the U-Net backbone")
        if any(p % 32 for p in patch_size):
            raise ValueError(f"Swin UNETR needs inputs divisible by 32, got {patch_size}")
        self.task_id = task_id
        self.task_type = self._get_task_type(task_id)
        if self.task_type == "segmentation":
            self.backbone = SwinBackbone(input_channels, 2, "segmentation")
            self.head = None
        elif self.task_type == "classification":
            self.backbone = SwinBackbone(input_channels, 1, "encoder")
            self.head = ClsRegHead(in_channels=SWIN_DEEPEST, num_classes=2)
        elif self.task_type == "regression":
            self.backbone = SwinBackbone(input_channels, 1, "encoder")
            self.head = ClsRegHead(in_channels=SWIN_DEEPEST, num_classes=1)
        else:
            raise ValueError(f"Unknown task type {self.task_type!r} for task_id {task_id}")

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        for p in self.backbone.parameters():
            p.requires_grad = False

        if lora_r and lora_r > 0:
            try:
                from src.lora import _replace_linear_with_lora, _replace_conv3d_with_lora
            except ImportError:
                from lora import _replace_linear_with_lora, _replace_conv3d_with_lora
            n_lin = _replace_linear_with_lora(self.backbone.encoder, "", lora_r, lora_alpha, lora_dropout)
            n_conv = 0
            if lora_decoder and not isinstance(self.backbone.decoder, nn.Identity):
                n_conv = _replace_conv3d_with_lora(self.backbone.decoder, "", lora_r, lora_alpha, lora_dropout)
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
            print(f"Injected LoRA into {n_lin} Linear + {n_conv} Conv3d layers (r={lora_r})")

        if self.head is not None:
            for p in self.head.parameters():
                p.requires_grad = True

    @staticmethod
    def _get_task_type(task_id: int) -> str:
        try:
            from src.data import TASK_CONFIGS
        except ImportError:
            from data import TASK_CONFIGS
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {task_id}")
        return TASK_CONFIGS[task_id]["task_type"]

    def _load_pretrained(self, path: str):
        """Load the self-supervised swinViT weights. The checkpoint holds the encoder's own
        state dict under 'module.' (MONAI's SSLHead); the 1-channel CT patch embedding is
        skipped when the task has a different channel count, as the U-Net's input
        convolution is."""
        ck = torch.load(path, map_location="cpu", weights_only=True)
        sd = ck.get("state_dict", ck)
        enc_sd = self.backbone.encoder.state_dict()
        load_sd = {}
        for k, v in sd.items():
            kk = k
            for pre in ("module.", "swinViT."):
                if kk.startswith(pre):
                    kk = kk[len(pre):]
            # the checkpoint predates MONAI's renaming of the MLP layers (fc1/fc2 -> linear1/linear2)
            kk = kk.replace(".mlp.fc1.", ".mlp.linear1.").replace(".mlp.fc2.", ".mlp.linear2.")
            if kk in enc_sd and enc_sd[kk].shape == v.shape:
                load_sd[kk] = v
        self.backbone.encoder.load_state_dict(load_sd, strict=False)
        print(f"Loaded {len(load_sd)}/{len(enc_sd)} pretrained swinViT keys from {Path(path).name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task_type == "segmentation":
            return self.backbone(x)
        return self.head(self.backbone(x))
