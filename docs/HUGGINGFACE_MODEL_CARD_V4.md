---
license: mit
library_name: pytorch
tags:
  - text-to-speech
  - speaker-cloning
  - speech-restoration
  - cosyvoice
  - narrowband
---

# Robust Speaker Cloning V4 Prompt Feature Restorer

This checkpoint is a 161,361-parameter residual temporal restorer for the exact 80-D acoustic prompt feature consumed by CosyVoice Flow. It does **not** replace or fine-tune CAMPPlus, the speech tokenizer, CosyVoice LLM, Flow or vocoder.

## Intended use

Use only behind the repository's locked narrowband router. The router computes the proportion of STFT energy above 4.2 kHz and applies the restorer only below the selection-derived threshold `7.893876957903083e-05`. Other inputs must use `effective_alpha=0` and remain on the official CosyVoice path.

Do not force the checkpoint onto arbitrary noise, reverberation or codec inputs. The all-degradation 105-case run was negative overall.

## Training

- Base speech: AISHELL-1.
- 400 speaker-disjoint train/validation speakers.
- 2,000 clean prompts × 7 degradations = 14,000 paired examples.
- Exact CosyVoice Matcha features: 24 kHz, 80 mel bins, n_fft/win 1920, hop 480, fmax 8 kHz.
- Real MUSAN noise, measured RIR, ffmpeg MP3/Opus and telephone narrowband corruption.
- 25 epochs; best validation total loss at epoch 25.

## Evaluation

Fresh-random-seed confirmation: 105 paired generations, 7 degradations × 15, 40 held-out speakers.

| Metric | Baseline | Routed V4 | Delta |
|---|---:|---:|---:|
| ECAPA cosine | 0.61677 | 0.62233 | +0.00556 |
| UTMOS22 strong | 2.63001 | 2.65358 | +0.02357 |

ECAPA 95% bootstrap CI `[+0.00174,+0.00998]`; paired two-sided Wilcoxon `p=0.01759`. The route changed 15/105 cases and strictly bypassed 90/105.

## Limitations

- Evidence is from one Mandarin corpus and one held-out speaker pool.
- The confirmation uses new corruptions and sample IDs but shares the same 40 held-out speakers with the earlier test.
- The router missed one telephone case and triggered one Opus case.
- No human MOS/ABX or cross-language/device validation is available.
- This model must not be described as universally improving noisy speaker cloning or as outperforming CAMPPlus.

See the repository's `claim_evidence_table.md` and `docs/AUTODL_V4_PROMPT_FEATURE_REPORT_2026-09-03.md` for the full positive and negative evidence.
