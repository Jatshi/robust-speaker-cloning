import numpy as np
import pandas as pd
import torch

from src.eval_protocol import build_eval_protocol
from src.metrics import mel_lsd_db, paired_summary
from src.models.robust_speaker_encoder import apply_soft_bwe, soft_bwe_gate
from src.quality import estimate_quality, perturb_quality
from src.splits import split_speakers


def test_soft_bwe_routing_prefers_narrow_band_inputs():
    gate = soft_bwe_gate(torch.tensor([3400 / 8000, 7000 / 8000]))
    assert gate[0] > 0.8
    assert gate[1] < 0.1
    original, predicted = torch.zeros(2, 80, 8), torch.ones(2, 80, 8)
    mixed, returned = apply_soft_bwe(original, predicted, torch.tensor([[0.5, 3400 / 8000], [0.5, 7000 / 8000]]))
    assert torch.allclose(mixed.mean((1, 2)), returned)


def test_quality_estimator_and_training_perturbation_are_bounded():
    waveform = torch.sin(2 * torch.pi * 440 * torch.arange(16000) / 16000)
    quality, details = estimate_quality(waveform)
    assert quality.shape == (2,) and torch.all((quality >= 0) & (quality <= 1))
    assert details["estimated_bandwidth_hz"] > 0
    perturbed = perturb_quality(quality[None], (0.1, 0.1), training=True)
    assert torch.all((perturbed >= 0) & (perturbed <= 1))


def test_mel_lsd_is_zero_only_for_equal_inputs():
    clean = np.zeros((80, 20))
    assert mel_lsd_db(clean, clean) == 0.0
    assert mel_lsd_db(clean + 0.2, clean) > 0.0


def test_paired_summary_reports_statistics_per_degradation():
    frame = pd.DataFrame({
        "degradation": ["noise"] * 4 + ["telephone"] * 4,
        "baseline_xvector_similarity": [0.3] * 8,
        "robust_xvector_similarity": [0.4, 0.5, 0.4, 0.5, 0.2, 0.35, 0.4, 0.45],
    })
    summary = paired_summary(frame)
    assert summary["comparisons"][0]["n"] == 8
    assert {row["group"] for row in summary["comparisons"]} == {"overall", "noise", "telephone"}


def test_eval_protocol_separates_prompt_reference_and_synthesis_text():
    manifest = {
        "S1": [
            {"audio_id": "a", "wav_path": "a.wav", "text": "提示句"},
            {"audio_id": "b", "wav_path": "b.wav", "text": "参考句"},
        ]
    }
    rows = build_eval_protocol(manifest, samples_per_type=2)
    assert len(rows) == 14
    assert all(row["prompt_audio_id"] != row["reference_audio_id"] for row in rows)
    assert all(row["prompt_text"] != row["synthesis_text"] for row in rows)


def test_train_selection_and_test_speakers_are_disjoint():
    splits = split_speakers([f"S{index:03d}" for index in range(100)])
    assert {key: len(value) for key, value in splits.items()} == {"train": 85, "selection": 5, "test": 10}
    assert not splits["train"] & splits["selection"]
    assert not splits["train"] & splits["test"]
    assert not splits["selection"] & splits["test"]
