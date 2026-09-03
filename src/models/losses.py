"""Losses for interface distillation, speaker identity and mel BWE."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    labels: torch.Tensor | None = None,
    negatives: torch.Tensor | None = None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """In-batch InfoNCE with an optional detached memory-bank of negatives."""
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.transpose(0, 1) / temperature
    if labels is None:
        labels = torch.arange(anchor.shape[0], device=anchor.device)
    if negatives is not None and negatives.numel():
        logits = torch.cat([logits, anchor @ F.normalize(negatives.detach(), dim=-1).T / temperature], dim=1)
    return F.cross_entropy(logits, labels.long())


def combined_loss(
    anchor: torch.Tensor,
    clean_embedding: torch.Tensor,
    teacher: torch.Tensor,
    labels: torch.Tensor,
    bwe_weight: torch.Tensor,
    predicted_mel: torch.Tensor,
    clean_mel: torch.Tensor,
    *,
    negatives: torch.Tensor | None = None,
    info_nce_weight: float = 1.0,
    distill_weight: float = 1.0,
    bwe_l1_weight: float = 0.5,
    bwe_mse_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Return named components so every optimization claim remains auditable.

    ``bwe_weight`` accepts either the legacy boolean telephone mask or the new
    continuous soft-routing weight.
    """
    teacher = F.normalize(teacher.reshape_as(anchor), dim=-1)
    contrastive = info_nce_loss(anchor, clean_embedding, labels, negatives=negatives)
    distill = (1.0 - F.cosine_similarity(anchor, teacher, dim=-1)).mean()
    route = bwe_weight.to(predicted_mel.dtype).reshape(-1, 1, 1)
    denominator = route.sum().clamp_min(1.0) * predicted_mel.shape[1] * predicted_mel.shape[2]
    difference = predicted_mel - clean_mel
    bwe_l1 = (difference.abs() * route).sum() / denominator
    bwe_mse = (difference.square() * route).sum() / denominator
    total = (
        info_nce_weight * contrastive
        + distill_weight * distill
        + bwe_l1_weight * bwe_l1
        + bwe_mse_weight * bwe_mse
    )
    return {"total": total, "info_nce": contrastive, "distill": distill, "bwe_l1": bwe_l1, "bwe_mse": bwe_mse}
