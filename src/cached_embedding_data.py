"""Memory-mapped V4 raw CAMPPlus training pairs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedCampPlusPairs(Dataset):
    def __init__(self, cache_dir: Path, selected_speakers: set[str]) -> None:
        self.meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        count, variants, dimension = self.meta["count"], self.meta["variants"], self.meta["dimension"]
        self.clean = np.memmap(cache_dir / "clean_embeddings.f32", np.float32, "r", shape=(count, dimension))
        self.degraded = np.memmap(
            cache_dir / "degraded_embeddings.f32", np.float32, "r", shape=(count * variants, dimension)
        )
        self.quality = np.memmap(cache_dir / "quality.f32", np.float32, "r", shape=(count * variants, 2))
        self.variants = int(variants)
        self.sources = [index for index, speaker in enumerate(self.meta["speakers"]) if speaker in selected_speakers]
        if not self.sources:
            raise ValueError("no cached rows match selected speakers")
        prototype_accumulator: dict[str, list[np.ndarray]] = {}
        for index in self.sources:
            prototype_accumulator.setdefault(self.meta["speakers"][index], []).append(np.asarray(self.clean[index]))
        self.prototypes = {
            speaker: np.stack(values).mean(axis=0).astype(np.float32) for speaker, values in prototype_accumulator.items()
        }

    def __len__(self) -> int:
        return len(self.sources) * self.variants

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source = self.sources[index // self.variants]
        variant = index % self.variants
        speaker = self.meta["speakers"][source]
        cached_index = source * self.variants + variant
        return {
            "degraded": torch.from_numpy(np.asarray(self.degraded[cached_index]).copy()),
            "clean": torch.from_numpy(np.asarray(self.clean[source]).copy()),
            "prototype": torch.from_numpy(self.prototypes[speaker].copy()),
            "quality": torch.from_numpy(np.asarray(self.quality[cached_index]).copy()),
        }


def clean_statistics(cache_dir: Path, selected_speakers: set[str]) -> tuple[torch.Tensor, torch.Tensor]:
    meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    clean = np.memmap(
        cache_dir / "clean_embeddings.f32", np.float32, "r", shape=(meta["count"], meta["dimension"])
    )
    indices = [index for index, speaker in enumerate(meta["speakers"]) if speaker in selected_speakers]
    values = torch.from_numpy(np.asarray(clean[indices]).copy())
    return values.mean(dim=0), values.std(dim=0, unbiased=False).clamp_min(1e-5)
