"""EDM network construction for 70x70 OpenFWI velocity fields."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


from .vendor.precond import EDMPrecond


class EDM70Precond(nn.Module):
    """EDM preconditioning with replication padding from 70 to 72 grid cells."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.img_resolution = 70
        self.img_channels = int(kwargs.get("img_channels", 1))
        self.sigma_min = float(kwargs.get("sigma_min", 0.002))
        self.sigma_max = float(kwargs.get("sigma_max", 80.0))
        self.sigma_data = float(kwargs.get("sigma_data", 0.5))
        internal_kwargs = dict(kwargs)
        internal_kwargs["img_resolution"] = 72
        internal_kwargs["use_fp16"] = bool(
            internal_kwargs.get("use_fp16", torch.cuda.is_available())
        )
        self.model = EDMPrecond(**internal_kwargs)

    def forward(
        self, x: torch.Tensor, sigma: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        if x.shape[-2:] == (70, 70):
            padded = F.pad(x.float(), (1, 1, 1, 1), mode="replicate")
            output = self.model(padded, sigma, *args, **kwargs)
            return output[..., 1:-1, 1:-1]
        if x.shape[-2:] != (72, 72):
            raise ValueError(
                f"expected 70x70 or 72x72 input, got {tuple(x.shape[-2:])}"
            )
        return self.model(x, sigma, *args, **kwargs)

    def round_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.model.round_sigma(sigma)


def build_edm_model(model_cfg: dict[str, Any]) -> EDM70Precond:
    """Build an EDM prior with a 70x70 external input/output grid."""
    if int(model_cfg.get("img_resolution", 70)) != 70:
        raise ValueError("OpenFWI models require img_resolution=70")
    return EDM70Precond(
        img_resolution=72,
        img_channels=int(model_cfg.get("img_channels", 1)),
        label_dim=int(model_cfg.get("label_dim", 0)),
        sigma_min=float(model_cfg.get("sigma_min", 0.002)),
        sigma_max=float(model_cfg.get("sigma_max", 80.0)),
        sigma_data=float(model_cfg.get("sigma_data", 0.5)),
        use_fp16=bool(model_cfg.get("use_fp16", torch.cuda.is_available())),
        model_type=str(model_cfg.get("model_type", "DhariwalUNet")),
        model_channels=int(model_cfg.get("model_channels", 128)),
        channel_mult=[int(x) for x in model_cfg.get("channel_mult", [1, 2, 3, 4])],
        channel_mult_emb=int(model_cfg.get("channel_mult_emb", 4)),
        num_blocks=int(model_cfg.get("num_blocks", 2)),
        attn_resolutions=[
            int(x) for x in model_cfg.get("attn_resolutions", [35, 17, 8])
        ],
        dropout=float(model_cfg.get("dropout", 0.0)),
        label_dropout=float(model_cfg.get("label_dropout", 0.0)),
    )


def edm_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
