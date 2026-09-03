"""Memory-mapped paired CosyVoice prompt features."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedPromptFeatures(Dataset):
    def __init__(self, root: Path, selected_speakers: set[str]) -> None:
        self.meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        count, variants, mels, frames = (self.meta[key] for key in ("count", "variants", "mels", "frames"))
        self.clean = np.memmap(root / "clean_features.f16", np.float16, "r", shape=(count, mels, frames))
        self.degraded = np.memmap(root / "degraded_features.f16", np.float16, "r", shape=(count * variants, mels, frames))
        self.quality = np.memmap(root / "quality.f32", np.float32, "r", shape=(count * variants, 2))
        self.variants = int(variants)
        self.sources = [index for index, speaker in enumerate(self.meta["speakers"]) if speaker in selected_speakers]
        if not self.sources:
            raise ValueError("no prompt features match the selected speakers")

    def __len__(self) -> int:
        return len(self.sources) * self.variants

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source, variant = self.sources[index // self.variants], index % self.variants
        cached = source * self.variants + variant
        return {
            "clean": torch.from_numpy(np.asarray(self.clean[source], dtype=np.float32)),
            "degraded": torch.from_numpy(np.asarray(self.degraded[cached], dtype=np.float32)),
            "quality": torch.from_numpy(np.asarray(self.quality[cached]).copy()),
        }


def prompt_feature_statistics(root: Path, selected_speakers: set[str]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    clean = np.memmap(root / "clean_features.f16", np.float16, "r", shape=(meta["count"], meta["mels"], meta["frames"]))
    indices = [index for index, speaker in enumerate(meta["speakers"]) if speaker in selected_speakers]
    values = torch.from_numpy(np.asarray(clean[indices], dtype=np.float32))
    return values.mean(dim=(0, 2)), values.std(dim=(0, 2), unbiased=False).clamp_min(1e-5)
