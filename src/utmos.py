"""Pinned UTMOS-compatible scoring adapter.

The official UTMOS22 repository requires a larger Hydra/fairseq pipeline.
For batch evaluation we use SpeechMOS v1.2.0's public UTMOS22-strong
reimplementation and record that implementation name in every result.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torchaudio


class UTMOS22StrongScorer:
    implementation = "tarepan/SpeechMOS:v1.2.0:utmos22_strong"

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        if hasattr(self.predictor, "to"):
            self.predictor = self.predictor.to(device)

    @torch.inference_mode()
    def score(self, path: Path) -> float:
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True).to(self.device)
        value = self.predictor(waveform, sample_rate)
        if isinstance(value, torch.Tensor):
            return float(value.float().mean().cpu())
        return float(value)
