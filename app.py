"""V2 Gradio 演示：有损参考 -> 鲁棒条件 -> 真实 CosyVoice2 合成。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import torch
import torchaudio

from src.cosyvoice_infer import CosyVoiceConditionedSynthesizer
from src.models.robust_speaker_encoder import LightweightBWENet, RobustSpeakerEncoder

ROOT = Path(__file__).parent; DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = ROOT / "outputs" / "checkpoints" / "best.pt"; MODEL_ROOT = ROOT / "models" / "CosyVoice2-0.5B"; COSYVOICE_ROOT = ROOT / "CosyVoice"
MEL = torchaudio.transforms.MelSpectrogram(16000, n_fft=512, win_length=512, hop_length=256, n_mels=80)


def _load_models() -> tuple[RobustSpeakerEncoder, LightweightBWENet]:
    if not CHECKPOINT.exists(): raise gr.Error("V2 正式训练尚未完成，未找到 best.pt。")
    state = torch.load(CHECKPOINT, map_location=DEVICE); encoder, bwe = RobustSpeakerEncoder().to(DEVICE).eval(), LightweightBWENet().to(DEVICE).eval()
    encoder.load_state_dict(state["encoder"]); bwe.load_state_dict(state["bwe"]); return encoder, bwe


def _quality(waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, bool, str]:
    spectrum = torch.fft.rfft(waveform).abs().square(); cumulative = torch.cumsum(spectrum, 0) / spectrum.sum().clamp_min(1e-6)
    bandwidth = float(torch.searchsorted(cumulative, torch.tensor(.95)).item() * sample_rate / (2 * len(spectrum))); snr = float(20 * torch.log10(waveform.pow(2).mean().sqrt() / waveform.diff().pow(2).mean().sqrt().clamp_min(1e-6) + 1e-6) + 25)
    telephone = bandwidth < 4000; tier = "低质量" if snr < 10 or telephone else "中质量" if snr < 20 else "高质量"
    return torch.tensor([[max(0., min(1., snr / 30)), max(0., min(1., bandwidth / 8000))]], device=DEVICE), telephone, f"{tier}：SNR≈{snr:.1f} dB，带宽≈{bandwidth:.0f} Hz"


def generate(reference: str, prompt_text: str, text: str) -> tuple[str, str]:
    if not reference or not prompt_text or not text: raise gr.Error("请提供参考音频、参考转写和待合成文本。")
    waveform, sample_rate = torchaudio.load(reference); waveform = waveform.mean(dim=0)
    if sample_rate != 16000: waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    quality, telephone, report = _quality(waveform, 16000); mel = torch.log(MEL(waveform).clamp_min(1e-6))[None].to(DEVICE)
    encoder, bwe = _load_models()
    with torch.no_grad(): condition = encoder(bwe(mel) if telephone else mel, quality)[0]
    output = Path(tempfile.mkstemp(suffix=".wav")[1]); CosyVoiceConditionedSynthesizer(MODEL_ROOT, COSYVOICE_ROOT).synthesize_with_condition(text, prompt_text, Path(reference), condition, output)
    return str(output), report + ("；已启用 BWE-mel 路由。" if telephone else "；使用全频/中频鲁棒条件路由。")


with gr.Blocks(title="鲁棒说话人克隆 V2") as demo:
    gr.Markdown("# 鲁棒说话人克隆 V2\n上传压缩、电话或噪声参考音频；V2 将使用 Transformer 鲁棒条件并接入真实 CosyVoice2。")
    reference = gr.Audio(type="filepath", label="有损参考音频"); prompt = gr.Textbox(label="参考音频准确转写"); text = gr.Textbox(label="待合成文本", value="这是鲁棒说话人克隆 V2 的端到端演示。")
    button = gr.Button("生成"); output = gr.Audio(label="CosyVoice2 输出"); report = gr.Textbox(label="质量路由")
    button.click(generate, [reference, prompt, text], [output, report])

if __name__ == "__main__": demo.launch(server_name="0.0.0.0", server_port=7861)
