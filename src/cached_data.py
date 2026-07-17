"""从 V2 紧凑 mel 缓存高速加载 GPU 训练 batch。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class CachedRobustDataset(Dataset):
    def __init__(self, cache_dir: Path, teacher_cache: Path, manifest: Path, selected_speakers: set[str]) -> None:
        meta = json.loads((cache_dir / "metadata.json").read_text()); self.count, self.variants, self.frames, self.mels = meta["count"], meta["variants"], meta["frames"], meta["mels"]
        self.clean = np.memmap(cache_dir / "clean_mels.f16", np.float16, "r", shape=(self.count, self.mels, self.frames)); self.degraded = np.memmap(cache_dir / "degraded_mels.f16", np.float16, "r", shape=(self.count * self.variants, self.mels, self.frames)); self.quality = np.memmap(cache_dir / "quality.f32", np.float32, "r", shape=(self.count * self.variants, 2)); self.telephone = np.memmap(cache_dir / "telephone.bool", np.bool_, "r", shape=(self.count * self.variants,))
        manifest_rows: dict[str, list[dict]] = json.loads(manifest.read_text()); paths = [item["wav_path"] for speaker, rows in manifest_rows.items() for item in rows]; teacher = torch.load(teacher_cache, map_location="cpu")
        self.sources = [index for index, speaker in enumerate(meta["speakers"]) if speaker in selected_speakers]; self.teachers = teacher; self.paths = paths; self.source_speakers = meta["speakers"]; self.speaker_sources: dict[str, list[int]] = {}
        for index in self.sources: self.speaker_sources.setdefault(self.source_speakers[index], []).append(index)

    def __len__(self) -> int: return len(self.sources) * self.variants
    def __getitem__(self, index: int) -> dict:
        source, variant = self.sources[index // self.variants], index % self.variants; cached_index = source * self.variants + variant
        return {"degraded_mel": torch.from_numpy(self.degraded[cached_index].astype(np.float32)), "clean_mel": torch.from_numpy(self.clean[source].astype(np.float32)), "teacher": self.teachers[self.paths[source]], "quality": torch.from_numpy(self.quality[cached_index].copy()), "telephone": bool(self.telephone[cached_index])}


class CachedUniqueSpeakerSampler(Sampler[list[int]]):
    def __init__(self, dataset: CachedRobustDataset, batch_size: int, seed: int = 42) -> None: self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, 0
    def set_epoch(self, epoch: int) -> None: self.epoch = epoch
    def __len__(self) -> int: return len(self.dataset) // self.batch_size
    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch); speakers = list(self.dataset.speaker_sources); order = torch.randperm(len(speakers), generator=generator).tolist(); pointer = 0
        source_to_relative = {source: index for index, source in enumerate(self.dataset.sources)}
        for _ in range(len(self)):
            selected = [speakers[order[(pointer + offset) % len(order)]] for offset in range(self.batch_size)]; pointer += self.batch_size; batch = []
            for speaker in selected:
                source = self.dataset.speaker_sources[speaker][torch.randint(len(self.dataset.speaker_sources[speaker]), (1,), generator=generator).item()]; variant = torch.randint(self.dataset.variants, (1,), generator=generator).item(); batch.append(source_to_relative[source] * self.dataset.variants + variant)
            yield batch


def collate(batch: list[dict]) -> dict: return {key: torch.stack([row[key] for row in batch]) for key in ("degraded_mel", "clean_mel", "teacher", "quality")} | {"telephone": torch.tensor([row["telephone"] for row in batch], dtype=torch.bool)}
