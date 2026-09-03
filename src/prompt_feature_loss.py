"""Losses for exact CosyVoice prompt-feature restoration."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def restoration_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    target_z: torch.Tensor,
    clean_output: torch.Tensor,
    clean_input: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    coordinate_l1 = F.smooth_l1_loss(diagnostics["output_z"], target_z)
    coordinate_mse = F.mse_loss(diagnostics["output_z"], target_z)
    temporal = F.smooth_l1_loss(predicted[..., 1:] - predicted[..., :-1], target[..., 1:] - target[..., :-1])
    spectral = F.smooth_l1_loss(predicted[:, 1:] - predicted[:, :-1], target[:, 1:] - target[:, :-1])
    clean_identity = F.smooth_l1_loss(clean_output, clean_input)
    trust_region = (diagnostics["output_z"] - diagnostics["input_z"]).square().mean()
    total = coordinate_l1 + 0.25 * coordinate_mse + 0.20 * temporal + 0.10 * spectral + 0.25 * clean_identity + 0.01 * trust_region
    return {
        "total": total,
        "coordinate_l1": coordinate_l1,
        "coordinate_mse": coordinate_mse,
        "temporal": temporal,
        "spectral": spectral,
        "clean_identity": clean_identity,
        "trust_region": trust_region,
    }
