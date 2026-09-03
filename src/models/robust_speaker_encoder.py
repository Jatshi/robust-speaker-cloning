"""Quality-conditioned speaker encoder and mel bandwidth-extension network."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RobustSpeakerEncoder(nn.Module):
    """Map an 80-bin log-mel prompt to CosyVoice's 192-D speaker space.

    ``quality`` is a continuous two-value condition: estimated SNR/30 and
    bandwidth/8000. Calling it a discrete quality token would be inaccurate.
    """

    def __init__(
        self,
        n_mels: int = 80,
        embedding_dim: int = 192,
        model_dim: int = 288,
        layers: int = 6,
        heads: int = 6,
        feedforward_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Conv1d(n_mels, model_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, model_dim),
            nn.GELU(),
        )
        self.quality_projection = nn.Sequential(nn.Linear(2, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim))
        block = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers, norm=nn.LayerNorm(model_dim))
        self.attention_pool = nn.Linear(model_dim, 1)
        self.output_projection = nn.Sequential(nn.LayerNorm(model_dim), nn.Linear(model_dim, embedding_dim))

    def forward(self, mel: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3 or mel.shape[1] != self.input_projection[0].in_channels:
            raise ValueError(f"mel must have shape [batch, 80, frames], got {tuple(mel.shape)}")
        if quality.shape != (mel.shape[0], 2):
            raise ValueError(f"quality must have shape [batch, 2], got {tuple(quality.shape)}")
        sequence = self.input_projection(mel).transpose(1, 2)
        sequence = sequence + self.quality_projection(quality.clamp(0.0, 1.0)).unsqueeze(1)
        sequence = self.transformer(sequence)
        weights = torch.softmax(self.attention_pool(sequence).squeeze(-1), dim=-1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return F.normalize(self.output_projection(pooled), dim=-1)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class LightweightBWENet(nn.Module):
    """Residual 1-D U-Net operating on log-mel sequences.

    It predicts a correction rather than a waveform. Consequently any LSD
    reported for this module must be labelled *mel-domain LSD*.
    """

    def __init__(self, n_mels: int = 80, widths: tuple[int, int, int] = (128, 256, 384)) -> None:
        super().__init__()
        first, second, third = widths
        self.encoder1 = _ConvBlock(n_mels, first)
        self.encoder2 = _ConvBlock(first, second)
        self.encoder3 = _ConvBlock(second, third)
        self.bottleneck = _ConvBlock(third, third)
        self.decoder2 = _ConvBlock(third + second, second)
        self.decoder1 = _ConvBlock(second + first, first)
        self.output = nn.Conv1d(first, n_mels, kernel_size=1)

    @staticmethod
    def _resize(value: torch.Tensor, frames: int) -> torch.Tensor:
        return F.interpolate(value, size=frames, mode="linear", align_corners=False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim != 3:
            raise ValueError(f"mel must have shape [batch, mels, frames], got {tuple(mel.shape)}")
        first = self.encoder1(mel)
        second = self.encoder2(F.avg_pool1d(first, kernel_size=2, ceil_mode=True))
        third = self.encoder3(F.avg_pool1d(second, kernel_size=2, ceil_mode=True))
        latent = self.bottleneck(third)
        decoded2 = self.decoder2(torch.cat([self._resize(latent, second.shape[-1]), second], dim=1))
        decoded1 = self.decoder1(torch.cat([self._resize(decoded2, first.shape[-1]), first], dim=1))
        return mel + self.output(decoded1)


def soft_bwe_gate(
    bandwidth_normalized: torch.Tensor,
    threshold_hz: float = 4200.0,
    temperature_hz: float = 400.0,
) -> torch.Tensor:
    """Return a smooth BWE mixture weight from a normalized bandwidth estimate."""
    bandwidth_hz = bandwidth_normalized.clamp(0.0, 1.0) * 8000.0
    return torch.sigmoid((threshold_hz - bandwidth_hz) / temperature_hz)


def apply_soft_bwe(
    degraded_mel: torch.Tensor,
    predicted_mel: torch.Tensor,
    quality: torch.Tensor,
    threshold_hz: float = 4200.0,
    temperature_hz: float = 400.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend original and BWE mel continuously, avoiding a brittle boolean route."""
    gate = soft_bwe_gate(quality[:, 1], threshold_hz, temperature_hz)
    enhanced = torch.lerp(degraded_mel, predicted_mel, gate[:, None, None])
    return enhanced, gate
