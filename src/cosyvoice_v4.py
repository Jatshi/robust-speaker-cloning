"""Scale-correct and deterministic CosyVoice interface for V4."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CosyVoiceNativeCampPlus:
    """Use the official raw CAMPPlus coordinates; never normalise on injection."""

    def __init__(self, model_root: Path, cosyvoice_root: Path) -> None:
        for directory in (cosyvoice_root, cosyvoice_root / "third_party" / "Matcha-TTS"):
            if str(directory) not in sys.path:
                sys.path.insert(0, str(directory))
        from cosyvoice.cli.cosyvoice import AutoModel  # pylint: disable=import-outside-toplevel
        from cosyvoice.utils.onnx import EmbeddingExtractor  # pylint: disable=import-outside-toplevel

        self.model = AutoModel(model_dir=str(model_root))
        self.extractor = EmbeddingExtractor(str(model_root / "campplus.onnx"))

    @staticmethod
    def load_prompt(path: Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.mean(dim=0)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        return waveform

    @torch.inference_mode()
    def extract_raw_embedding(self, waveform_or_path: torch.Tensor | Path) -> torch.Tensor:
        waveform = self.load_prompt(waveform_or_path) if isinstance(waveform_or_path, Path) else waveform_or_path
        if waveform.ndim == 1:
            waveform = waveform[None]
        embedding = self.extractor.inference(waveform.cpu()).float().reshape(-1)
        if embedding.numel() != 192:
            raise RuntimeError(f"unexpected CAMPPlus embedding shape: {tuple(embedding.shape)}")
        return embedding

    @torch.inference_mode()
    def extract_prompt_feature(self, prompt_wav: Path) -> torch.Tensor:
        feature, _ = self.model.frontend._extract_speech_feat(str(prompt_wav))  # pylint: disable=protected-access
        return feature.transpose(1, 2).float()

    def _write(self, pieces: list[torch.Tensor], output: Path) -> None:
        if not pieces:
            raise RuntimeError("CosyVoice did not return a waveform")
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, torch.cat(pieces).numpy(), self.model.sample_rate)

    @torch.inference_mode()
    def synthesize_baseline(
        self, text: str, prompt_text: str, prompt_wav: Path, output: Path, seed: int
    ) -> None:
        seed_everything(seed)
        pieces = [
            item["tts_speech"].squeeze().cpu()
            for item in self.model.inference_zero_shot(
                text, prompt_text, str(prompt_wav), stream=False, text_frontend=False
            )
        ]
        self._write(pieces, output)

    @torch.inference_mode()
    def synthesize_raw_embedding(
        self,
        text: str,
        prompt_text: str,
        prompt_wav: Path,
        raw_embedding: torch.Tensor,
        output: Path,
        seed: int,
    ) -> None:
        seed_everything(seed)
        condition = raw_embedding.reshape(1, 192).float().to(self.model.frontend.device)
        normalized_prompt = self.model.frontend.text_normalize(prompt_text, split=False, text_frontend=False)
        normalized_texts = self.model.frontend.text_normalize(text, split=True, text_frontend=False)
        pieces: list[torch.Tensor] = []
        for normalized_text in normalized_texts:
            model_input = self.model.frontend.frontend_zero_shot(
                normalized_text, normalized_prompt, str(prompt_wav), self.model.sample_rate, ""
            )
            model_input["llm_embedding"] = condition
            model_input["flow_embedding"] = condition
            pieces.extend(
                item["tts_speech"].squeeze().cpu()
                for item in self.model.model.tts(**model_input, stream=False, speed=1.0)
            )
        self._write(pieces, output)

    @torch.inference_mode()
    def synthesize_mixed_prompt(
        self,
        text: str,
        prompt_text: str,
        degraded_prompt_wav: Path,
        clean_prompt_wav: Path,
        clean_components: set[str],
        output: Path,
        seed: int,
    ) -> None:
        """Oracle-only component swap used to locate the actual bottleneck.

        ``clean_components`` may contain ``embedding``, ``tokens`` and/or
        ``features``.  This method is diagnostic: clean prompt components are
        unavailable in a real deployment and must never be reported as a
        deployable result.
        """
        allowed = {"embedding", "tokens", "features"}
        if not clean_components <= allowed:
            raise ValueError(f"unknown clean prompt components: {clean_components - allowed}")
        seed_everything(seed)
        normalized_prompt = self.model.frontend.text_normalize(prompt_text, split=False, text_frontend=False)
        normalized_texts = self.model.frontend.text_normalize(text, split=True, text_frontend=False)
        groups = {
            "embedding": ("llm_embedding", "flow_embedding"),
            "tokens": (
                "llm_prompt_speech_token",
                "llm_prompt_speech_token_len",
                "flow_prompt_speech_token",
                "flow_prompt_speech_token_len",
            ),
            "features": ("prompt_speech_feat", "prompt_speech_feat_len"),
        }
        pieces: list[torch.Tensor] = []
        for normalized_text in normalized_texts:
            degraded_input = self.model.frontend.frontend_zero_shot(
                normalized_text, normalized_prompt, str(degraded_prompt_wav), self.model.sample_rate, ""
            )
            clean_input = self.model.frontend.frontend_zero_shot(
                normalized_text, normalized_prompt, str(clean_prompt_wav), self.model.sample_rate, ""
            )
            for component in clean_components:
                for key in groups[component]:
                    degraded_input[key] = clean_input[key]
            pieces.extend(
                item["tts_speech"].squeeze().cpu()
                for item in self.model.model.tts(**degraded_input, stream=False, speed=1.0)
            )
        self._write(pieces, output)

    @torch.inference_mode()
    def synthesize_prompt_feature(
        self,
        text: str,
        prompt_text: str,
        prompt_wav: Path,
        prompt_feature: torch.Tensor,
        output: Path,
        seed: int,
    ) -> None:
        """Replace only Flow's acoustic prompt feature; keep tokens/embedding official."""
        seed_everything(seed)
        normalized_prompt = self.model.frontend.text_normalize(prompt_text, split=False, text_frontend=False)
        normalized_texts = self.model.frontend.text_normalize(text, split=True, text_frontend=False)
        feature = prompt_feature.reshape(1, 80, -1).transpose(1, 2).to(self.model.frontend.device)
        pieces: list[torch.Tensor] = []
        for normalized_text in normalized_texts:
            model_input = self.model.frontend.frontend_zero_shot(
                normalized_text, normalized_prompt, str(prompt_wav), self.model.sample_rate, ""
            )
            model_input["prompt_speech_feat"] = feature
            model_input["prompt_speech_feat_len"] = torch.tensor(
                [feature.shape[1]], dtype=torch.int32, device=self.model.frontend.device
            )
            pieces.extend(
                item["tts_speech"].squeeze().cpu()
                for item in self.model.model.tts(**model_input, stream=False, speed=1.0)
            )
        self._write(pieces, output)
