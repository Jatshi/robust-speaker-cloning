"""多正样本 InfoNCE、BWE 重建和 CosyVoice 条件蒸馏损失。"""
from __future__ import annotations

import torch
from torch.nn import functional as functional


def supervised_info_nce(anchor: torch.Tensor, positive: torch.Tensor, speaker_ids: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = anchor @ positive.T / temperature
    same_speaker = speaker_ids[:, None].eq(speaker_ids[None, :])
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_probability.masked_select(same_speaker).view(anchor.size(0), -1).mean(dim=1)).mean()


def combined_loss(anchor: torch.Tensor, clean_embedding: torch.Tensor, teacher: torch.Tensor, speaker_ids: torch.Tensor,
                  telephone_mask: torch.Tensor, predicted_mel: torch.Tensor, target_mel: torch.Tensor,
                  temperature: float = 0.07, bwe_weight: float = 0.3, distill_weight: float = 0.5) -> dict[str, torch.Tensor]:
    info_nce = supervised_info_nce(anchor, clean_embedding, speaker_ids, temperature)
    distill = 1 - functional.cosine_similarity(anchor, teacher, dim=-1).mean()
    if telephone_mask.any():
        prediction, target = predicted_mel[telephone_mask], target_mel[telephone_mask]
        bwe_l1 = functional.l1_loss(prediction, target)
        bwe_mse = functional.mse_loss(prediction, target)
    else:
        bwe_l1, bwe_mse = anchor.new_zeros(()), anchor.new_zeros(())
    total = info_nce + distill_weight * distill + bwe_weight * (bwe_l1 + bwe_mse)
    return {"total": total, "info_nce": info_nce, "distill": distill, "bwe_l1": bwe_l1, "bwe_mse": bwe_mse}
