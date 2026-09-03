"""Identity-initialised calibration in CosyVoice's native CAMPPlus space."""
from __future__ import annotations

import torch
from torch import nn


class CampPlusResidualAdapter(nn.Module):
    """Predict a bounded residual without replacing the pretrained encoder.

    The last residual layer is exactly zero at initialisation.  Therefore every
    input is returned unchanged before training and whenever ``alpha=0``.
    Embeddings are standardised internally but returned in the original raw
    CAMPPlus coordinate system expected by CosyVoice.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        embedding_dim: int = 192,
        hidden_dim: int = 384,
        dropout: float = 0.05,
        max_residual_z: float = 3.0,
    ) -> None:
        super().__init__()
        if mean.numel() != embedding_dim or std.numel() != embedding_dim:
            raise ValueError("CAMPPlus statistics must contain 192 coordinates")
        self.register_buffer("embedding_mean", mean.float().reshape(1, embedding_dim))
        self.register_buffer("embedding_std", std.float().clamp_min(1e-5).reshape(1, embedding_dim))
        self.max_residual_z = float(max_residual_z)
        input_dim = embedding_dim + 2
        self.residual = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        final = self.residual[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def standardise(self, embedding: torch.Tensor) -> torch.Tensor:
        return (embedding - self.embedding_mean) / self.embedding_std

    def destandardise(self, embedding: torch.Tensor) -> torch.Tensor:
        return embedding * self.embedding_std + self.embedding_mean

    def forward(
        self, embedding: torch.Tensor, quality: torch.Tensor, alpha: float | torch.Tensor = 1.0
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        embedding = embedding.reshape(-1, self.embedding_mean.shape[-1]).float()
        quality = quality.reshape(embedding.shape[0], 2).float().clamp(0.0, 1.0)
        z = self.standardise(embedding)
        features = torch.cat([z, quality], dim=-1)
        residual_z = torch.tanh(self.residual(features)) * self.max_residual_z
        gate = torch.sigmoid(self.gate(features))
        corrected_z = z + torch.as_tensor(alpha, device=z.device, dtype=z.dtype) * gate * residual_z
        corrected = self.destandardise(corrected_z)
        return corrected, {"gate": gate, "residual_z": residual_z, "input_z": z, "output_z": corrected_z}


def build_adapter_from_checkpoint(checkpoint: dict, device: torch.device) -> CampPlusResidualAdapter:
    config = checkpoint.get("model_config", {})
    model = CampPlusResidualAdapter(
        mean=checkpoint["embedding_mean"],
        std=checkpoint["embedding_std"],
        hidden_dim=int(config.get("hidden_dim", 384)),
        dropout=float(config.get("dropout", 0.05)),
        max_residual_z=float(config.get("max_residual_z", 3.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model
