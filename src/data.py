"""V2 在线退化数据集：42k 样本逻辑规模，不落盘复制退化 WAV。"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler

from src.degradation import DEGRADATION_TYPES, simulate


class OnlineRobustSpeakerDataset(Dataset):
    def __init__(self, manifest: Path, teacher_cache: Path, speaker_subset: set[str], segment_seconds: float = 4.0, seed: int = 42) -> None:
        all_speakers: dict[str, list[dict]] = json.loads(manifest.read_text(encoding="utf-8"))
        self.items = [(speaker, item) for speaker, entries in all_speakers.items() if speaker in speaker_subset for item in entries]
        self.teacher = torch.load(teacher_cache, map_location="cpu"); self.segment_samples = int(16000 * segment_seconds); self.seed = seed; self.epoch = 0
        self.mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=512, win_length=512, hop_length=256, n_mels=80)
        self.speaker_indices: dict[str, list[int]] = {}
        for index, (speaker, _) in enumerate(self.items): self.speaker_indices.setdefault(speaker, []).append(index)

    def set_epoch(self, epoch: int) -> None: self.epoch = epoch
    def __len__(self) -> int: return len(self.items) * len(DEGRADATION_TYPES)

    def _load_crop(self, path: str, seed: int) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path); waveform = waveform.mean(dim=0)
        if sample_rate != 16000: waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        if waveform.numel() > self.segment_samples:
            start = seed % (waveform.numel() - self.segment_samples + 1); waveform = waveform[start:start + self.segment_samples]
        return torch.nn.functional.pad(waveform, (0, max(0, self.segment_samples - waveform.numel())))

    def __getitem__(self, index: int) -> dict:
        source_index, type_index = divmod(index, len(DEGRADATION_TYPES)); speaker, item = self.items[source_index]
        deterministic_seed = self.seed + self.epoch * len(self) + index; clean = self._load_crop(item["wav_path"], deterministic_seed)
        degraded, metadata = simulate(clean, DEGRADATION_TYPES[type_index], deterministic_seed % 3, deterministic_seed)
        clean_mel = torch.log(self.mel(clean).clamp_min(1e-6)); degraded_mel = torch.log(self.mel(degraded).clamp_min(1e-6))
        quality = torch.tensor([float(metadata["snr_db"]) / 30.0, float(metadata["bandwidth_hz"]) / 8000.0])
        return {"degraded_mel": degraded_mel, "clean_mel": clean_mel, "teacher": self.teacher[item["wav_path"]],
                "quality": quality, "speaker": speaker, "telephone": bool(metadata["telephone"])}


class UniqueSpeakerBatchSampler(Sampler[list[int]]):
    """每个 batch 一个说话人一次，使 InfoNCE 的非对角线均是可靠负样本。"""
    def __init__(self, dataset: OnlineRobustSpeakerDataset, batch_size: int, seed: int = 42) -> None:
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, 0
    def set_epoch(self, epoch: int) -> None: self.epoch = epoch
    def __len__(self) -> int: return len(self.dataset) // self.batch_size
    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch); speakers = list(self.dataset.speaker_indices)
        order = torch.randperm(len(speakers), generator=generator).tolist(); pointer = 0
        for _ in range(len(self)):
            chosen = [speakers[order[(pointer + offset) % len(order)]] for offset in range(self.batch_size)]; pointer += self.batch_size
            yield [self.dataset.speaker_indices[speaker][torch.randint(len(self.dataset.speaker_indices[speaker]), (1,), generator=generator).item()] * len(DEGRADATION_TYPES) + torch.randint(len(DEGRADATION_TYPES), (1,), generator=generator).item() for speaker in chosen]


def collate(batch: list[dict]) -> dict:
    return {key: torch.stack([item[key] for item in batch]) for key in ("degraded_mel", "clean_mel", "teacher", "quality")} | {"telephone": torch.tensor([item["telephone"] for item in batch], dtype=torch.bool)}
