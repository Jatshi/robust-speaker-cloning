"""将 V2 的 192 维鲁棒条件接入真实 CosyVoice2 合成路径。"""
from __future__ import annotations

import sys
from pathlib import Path

import soundfile
import torch


class CosyVoiceConditionedSynthesizer:
    def __init__(self, model_root: Path, cosyvoice_root: Path) -> None:
        for directory in (cosyvoice_root, cosyvoice_root / "third_party" / "Matcha-TTS"):
            if str(directory) not in sys.path: sys.path.insert(0, str(directory))
        from cosyvoice.cli.cosyvoice import AutoModel  # pylint: disable=import-outside-toplevel
        self.model = AutoModel(model_dir=str(model_root))

    @torch.inference_mode()
    def synthesize_with_condition(self, text: str, prompt_text: str, prompt_wav: Path, condition: torch.Tensor, output: Path) -> None:
        condition = torch.nn.functional.normalize(condition.reshape(1, 192), dim=-1).to(self.model.frontend.device)
        normalized_prompt = self.model.frontend.text_normalize(prompt_text, split=False, text_frontend=False)
        normalized_texts = self.model.frontend.text_normalize(text, split=True, text_frontend=False)
        pieces = []
        for normalized_text in normalized_texts:
            model_input = self.model.frontend.frontend_zero_shot(normalized_text, normalized_prompt, str(prompt_wav), self.model.sample_rate, "")
            # The distillation target is CampPlus space; these are the two
            # fields consumed by the official CosyVoice2 model interface.
            model_input["llm_embedding"] = condition; model_input["flow_embedding"] = condition
            pieces.extend(item["tts_speech"].squeeze().cpu() for item in self.model.model.tts(**model_input, stream=False, speed=1.0))
        if not pieces: raise RuntimeError("CosyVoice did not return a waveform")
        output.parent.mkdir(parents=True, exist_ok=True); soundfile.write(output, torch.cat(pieces).numpy(), self.model.sample_rate)

    @torch.inference_mode()
    def synthesize_baseline(self, text: str, prompt_text: str, prompt_wav: Path, output: Path) -> None:
        pieces = [item["tts_speech"].squeeze().cpu() for item in self.model.inference_zero_shot(text, prompt_text, str(prompt_wav), stream=False, text_frontend=False)]
        if not pieces: raise RuntimeError("CosyVoice did not return a waveform")
        output.parent.mkdir(parents=True, exist_ok=True); soundfile.write(output, torch.cat(pieces).numpy(), self.model.sample_rate)
