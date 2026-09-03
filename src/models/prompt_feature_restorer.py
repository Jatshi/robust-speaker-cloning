"""Identity-initialised restoration of CosyVoice acoustic prompt features."""
from __future__ import annotations

import torch
from torch import nn


class _TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation, groups=channels),
            nn.Conv1d(channels, channels, 1),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class PromptFeatureRestorer(nn.Module):
    """Correct only the 80-D prompt feature consumed by CosyVoice Flow.

    CAMPPlus embeddings and speech tokens remain untouched.  The output layer
    is zero initialised, so the network starts as an exact no-op and alpha=0 is
    always a deployable fallback.
    """

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        channels: int = 128,
        blocks: int = 8,
        dropout: float = 0.05,
        max_residual_z: float = 4.0,
    ) -> None:
        super().__init__()
        self.register_buffer("feature_mean", mean.float().reshape(1, 80, 1))
        self.register_buffer("feature_std", std.float().clamp_min(1e-5).reshape(1, 80, 1))
        self.max_residual_z = float(max_residual_z)
        self.input = nn.Conv1d(82, channels, 1)
        dilations = (1, 2, 4, 8, 16)
        self.temporal = nn.Sequential(
            *[_TemporalResidualBlock(channels, dilations[index % len(dilations)], dropout) for index in range(blocks)]
        )
        self.output = nn.Conv1d(channels, 80, 1)
        self.gate = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def standardise(self, feature: torch.Tensor) -> torch.Tensor:
        return (feature - self.feature_mean) / self.feature_std

    def destandardise(self, feature: torch.Tensor) -> torch.Tensor:
        return feature * self.feature_std + self.feature_mean

    def forward(
        self, feature: torch.Tensor, quality: torch.Tensor, alpha: float | torch.Tensor = 1.0
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if feature.ndim != 3 or feature.shape[1] != 80:
            raise ValueError(f"feature must have shape [batch, 80, frames], got {tuple(feature.shape)}")
        quality = quality.reshape(feature.shape[0], 2).float().clamp(0.0, 1.0)
        z = self.standardise(feature.float())
        quality_frames = quality.unsqueeze(-1).expand(-1, -1, z.shape[-1])
        hidden = self.temporal(self.input(torch.cat([z, quality_frames], dim=1)))
        residual_z = torch.tanh(self.output(hidden)) * self.max_residual_z
        gate = torch.sigmoid(self.gate(quality)).reshape(-1, 1, 1)
        corrected_z = z + torch.as_tensor(alpha, device=z.device, dtype=z.dtype) * gate * residual_z
        return self.destandardise(corrected_z), {
            "input_z": z,
            "output_z": corrected_z,
            "residual_z": residual_z,
            "gate": gate,
        }


def build_feature_restorer_from_checkpoint(checkpoint: dict, device: torch.device) -> PromptFeatureRestorer:
    config = checkpoint.get("model_config", {})
    model = PromptFeatureRestorer(
        checkpoint["feature_mean"],
        checkpoint["feature_std"],
        channels=int(config.get("channels", 128)),
        blocks=int(config.get("blocks", 8)),
        dropout=float(config.get("dropout", 0.05)),
        max_residual_z=float(config.get("max_residual_z", 4.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model
