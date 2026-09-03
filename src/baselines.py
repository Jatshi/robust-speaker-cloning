"""Independent enhancement baseline used in the three-way evaluation."""
from __future__ import annotations

from pathlib import Path

import torch


class MetricGANPlusEnhancer:
    """SpeechBrain MetricGAN+ VoiceBank model; no project weights are reused."""

    def __init__(self, cache_dir: Path, device: torch.device) -> None:
        try:
            from speechbrain.inference.enhancement import SpectralMaskEnhancement
        except ModuleNotFoundError:
            from speechbrain.pretrained import SpectralMaskEnhancement
        self.model = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir=str(cache_dir),
            run_opts={"device": str(device)},
        )
        self.device = device

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        batch = waveform.reshape(1, -1).to(self.device)
        lengths = torch.ones(1, device=batch.device) if lengths is None else lengths
        enhanced = self.model.enhance_batch(batch, lengths=lengths)
        return enhanced.reshape(-1).detach().cpu()
