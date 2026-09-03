"""Losses that preserve CAMPPlus scale as well as speaker geometry."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def campplus_calibration_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    prototype: torch.Tensor,
    clean_output: torch.Tensor,
    clean_input: torch.Tensor,
    target_z: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    input_z, output_z = diagnostics["input_z"], diagnostics["output_z"]
    coordinate = F.smooth_l1_loss(output_z, target_z)
    cosine = (1.0 - F.cosine_similarity(predicted, target, dim=-1)).mean()
    prototype_cosine = (1.0 - F.cosine_similarity(predicted, prototype, dim=-1)).mean()
    predicted_norm = predicted.norm(dim=-1).clamp_min(1e-6)
    target_norm = target.norm(dim=-1).clamp_min(1e-6)
    norm = F.smooth_l1_loss(predicted_norm.log(), target_norm.log())
    clean_identity = F.smooth_l1_loss(clean_output, clean_input)
    trust_region = (output_z - input_z).square().mean()

    if predicted.shape[0] > 1:
        predicted_relation = F.normalize(predicted, dim=-1) @ F.normalize(predicted, dim=-1).T
        target_relation = F.normalize(target, dim=-1) @ F.normalize(target, dim=-1).T
        mask = ~torch.eye(predicted.shape[0], dtype=torch.bool, device=predicted.device)
        relational = F.smooth_l1_loss(predicted_relation[mask], target_relation[mask])
    else:
        relational = predicted.new_zeros(())

    total = (
        coordinate
        + 0.50 * cosine
        + 0.10 * norm
        + 0.10 * relational
        + 0.20 * prototype_cosine
        + 0.25 * clean_identity
        + 0.01 * trust_region
    )
    return {
        "total": total,
        "coordinate": coordinate,
        "cosine": cosine,
        "norm": norm,
        "relational": relational,
        "prototype": prototype_cosine,
        "clean_identity": clean_identity,
        "trust_region": trust_region,
    }
