"""深度 Transformer 说话人编码器与联合训练的 mel 域 BWE U-Net。"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, max_length: int = 1024) -> None:
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension))
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * divisor), torch.cos(position * divisor)
        self.register_buffer("encoding", encoding[None], persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.encoding[:, :value.size(1)]


class RobustSpeakerEncoder(nn.Module):
    """6 层 Transformer：mel + 质量 token -> 与 CosyVoice 兼容的 192 维条件。"""

    def __init__(self, n_mels: int = 80, d_model: int = 256, layers: int = 6, heads: int = 4,
                 feedforward: int = 1024, output_dim: int = 192, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.mel_projection = nn.Linear(n_mels, d_model)
        self.quality_projection = nn.Sequential(nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.position = PositionalEncoding(d_model)
        block = nn.TransformerEncoderLayer(d_model, heads, feedforward, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, layers)
        self.attention = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, output_dim))

    def forward(self, mel: torch.Tensor, quality: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if mel.ndim != 3:
            raise ValueError("mel must have shape (batch, 80, frames)")
        frames = mel.transpose(1, 2)
        values = self.position(self.mel_projection(frames) + self.quality_projection(quality).unsqueeze(1))
        padding_mask = None
        if lengths is not None:
            padding_mask = torch.arange(values.size(1), device=values.device)[None] >= lengths[:, None]
        values = self.transformer(values, src_key_padding_mask=padding_mask)
        weights = self.attention(values).squeeze(-1)
        if padding_mask is not None:
            weights = weights.masked_fill(padding_mask, float("-inf"))
        weights = weights.softmax(dim=-1)
        pooled = torch.sum(values * weights.unsqueeze(-1), dim=1)
        return functional.normalize(self.output(pooled), dim=-1)


class LightweightBWENet(nn.Module):
    """在低频 mel 条件下预测电话音频丢失的高频 mel 残差。"""

    def __init__(self, n_mels: int = 80, channels: int = 192) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(n_mels, channels, 5, padding=2), nn.GroupNorm(8, channels), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv1d(channels, channels * 2, 4, stride=2, padding=1), nn.GroupNorm(8, channels * 2), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv1d(channels * 2, channels * 2, 4, stride=2, padding=1), nn.GroupNorm(8, channels * 2), nn.SiLU())
        self.middle = nn.Sequential(nn.Conv1d(channels * 2, channels * 2, 3, padding=1), nn.SiLU(), nn.Conv1d(channels * 2, channels * 2, 3, padding=1), nn.SiLU())
        self.up2 = nn.ConvTranspose1d(channels * 2, channels * 2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.Conv1d(channels * 4, channels * 2, 3, padding=1), nn.GroupNorm(8, channels * 2), nn.SiLU())
        self.up1 = nn.ConvTranspose1d(channels * 2, channels, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.Conv1d(channels * 2, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.SiLU(), nn.Conv1d(channels, n_mels, 3, padding=1))

    @staticmethod
    def _match(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return value[..., :reference.size(-1)] if value.size(-1) >= reference.size(-1) else functional.pad(value, (0, reference.size(-1) - value.size(-1)))

    def forward(self, telephone_mel: torch.Tensor) -> torch.Tensor:
        first = self.enc1(telephone_mel); second = self.enc2(first); third = self.enc3(second)
        middle = self.middle(third); up2 = self._match(self.up2(middle), second)
        decoded2 = self.dec2(torch.cat((up2, second), dim=1)); up1 = self._match(self.up1(decoded2), first)
        residual = self.dec1(torch.cat((up1, first), dim=1))
        output = telephone_mel.clone(); output[:, output.size(1) // 2:] += residual[:, residual.size(1) // 2:]
        return output
