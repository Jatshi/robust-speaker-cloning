"""Shared objective evaluation helpers for V4."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio


def load_audio(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    return torchaudio.functional.resample(waveform, source_rate, sample_rate) if source_rate != sample_rate else waveform


def write_prompt(path: Path, waveform: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform.detach().cpu().numpy(), 16000)


def xvector(model, path: Path, device: torch.device) -> torch.Tensor:
    waveform = load_audio(path).reshape(1, -1).to(device)
    return torch.nn.functional.normalize(model.encode_batch(waveform)[0, 0], dim=0).cpu()


def waveform_mae(left: Path, right: Path) -> float:
    a, b = load_audio(left, sample_rate=22050), load_audio(right, sample_rate=22050)
    frames = min(a.numel(), b.numel())
    return float((a[:frames] - b[:frames]).abs().mean()) if frames else float("nan")


def load_protocol(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def speaker_model(output: Path, device: torch.device):
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ModuleNotFoundError:
        from speechbrain.pretrained import EncoderClassifier
    required = ("hyperparams.yaml", "embedding_model.ckpt", "mean_var_norm_emb.ckpt", "classifier.ckpt", "label_encoder.ckpt")
    local_candidates = (
        Path("models/speechbrain-ecapa"),
        Path("outputs/evaluation-v3/models/ecapa"),
    )
    local = next((candidate.resolve() for candidate in local_candidates if all((candidate / name).exists() for name in required)), None)
    if local is not None:
        print(json.dumps({"event": "ecapa_local_reuse", "source": str(local)}), flush=True)
        return EncoderClassifier.from_hparams(
            source=str(local),
            savedir=str(output / "models" / "ecapa"),
            overrides={"pretrained_path": str(local)},
            run_opts={"device": str(device)},
        ).eval()
    return EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(output / "models" / "ecapa"),
        run_opts={"device": str(device)},
    ).eval()
