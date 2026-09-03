from pathlib import Path

import soundfile as sf
import torch

from src.cosyvoice_infer import CosyVoiceConditionedSynthesizer


class _FakeFrontend:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def text_normalize(self, text, split, text_frontend):
        self.calls.append((text, split, text_frontend))
        return ["part-1", "part-2"] if split else "normalized-prompt"

    def frontend_zero_shot(self, text, prompt, prompt_wav, sample_rate, speaker_id):
        assert prompt == "normalized-prompt" and text.startswith("part-")
        return {"text_part": text, "llm_embedding": torch.zeros(1, 192), "flow_embedding": torch.zeros(1, 192)}


class _FakeCore:
    def __init__(self): self.inputs = []

    def tts(self, **model_input):
        self.inputs.append(model_input)
        yield {"tts_speech": torch.ones(1, 16)}


class _FakeCosyVoice:
    sample_rate = 24000

    def __init__(self): self.frontend, self.model = _FakeFrontend(), _FakeCore()


def test_condition_injection_preserves_official_text_path_and_both_embeddings(tmp_path: Path):
    synthesizer = CosyVoiceConditionedSynthesizer.__new__(CosyVoiceConditionedSynthesizer)
    synthesizer.model = _FakeCosyVoice(); output = tmp_path / "out.wav"
    synthesizer.synthesize_with_condition("raw text", "raw prompt", tmp_path / "prompt.wav", torch.randn(192), output)
    waveform, sample_rate = sf.read(output)
    assert sample_rate == 24000 and len(waveform) == 32
    assert len(synthesizer.model.model.inputs) == 2
    for row in synthesizer.model.model.inputs:
        assert row["llm_embedding"].shape == (1, 192)
        assert torch.equal(row["llm_embedding"], row["flow_embedding"])
