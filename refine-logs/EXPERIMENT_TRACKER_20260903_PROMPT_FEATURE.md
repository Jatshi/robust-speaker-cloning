# Robust Speaker Cloning V4 实验追踪

日期：2026-09-03  
AutoDL：RTX 4080 SUPER（平台报告约 32 GB 显存）

## 1. 已推翻的初始假设

最初怀疑 V3 把 CAMPPlus 从原始范数约 12.6 强制归一化到 1，破坏了下游分布。实际接口诊断显示 raw reinjection 与 unit-norm reinjection 相对 baseline 的 ECAPA 差和波形 MAE 都为 0；源码复核也发现 CosyVoice Flow 与 LLM 下游本来就执行 `F.normalize`。

结论：尺度假设被证伪，因此没有继续训练 CAMPPlus residual adapter，避免在错误方向烧 GPU。

## 2. 组件 Oracle

| 只替换的 clean 条件 | ECAPA mean delta | win rate | 95% CI |
|---|---:|---:|---:|
| embedding | +0.00275 | 5/7 | [-0.01920,+0.02033] |
| speech token | -0.01905 | 1/7 | [-0.03629,-0.00150] |
| prompt feature | +0.03534 | 7/7 | [+0.01150,+0.06446] |
| token + feature | +0.03596 | 5/7 | [-0.00224,+0.08175] |
| full clean | +0.04104 | 6/7 | [+0.00696,+0.08410] |

最强、最稳定的可利用 headroom 在 Flow 使用的 `prompt_speech_feat`，而不是 CAMPPlus。

## 3. 缓存与训练

- 400 speakers × 5 utterances = 2,000 clean prompts；7 种退化，共 14,000 pairs。
- 训练/验证 11,900/700，speaker-disjoint；cache 约 495 MB。
- 161,361 个可训练参数；25 epoch；单 epoch 约 2.4–3.7 s。
- 最佳 epoch 25，validation total `0.122310`。
- validation raw feature L1 `2.28309` → predicted `0.47711`。

这证明恢复器学会了逼近 clean feature，但不自动等于最终 TTS 身份提升。

## 4. Selection 与路由

14 条 selection 上 alpha 0.25/0.5/0.75/1 的总体 ECAPA delta 为 `+0.00289/+0.00634/+0.01388/+0.01515`。alpha=1 的 CI 仍跨 0，但收益集中于 telephone（2/2，mean `+0.11657`）。据此只用 selection 锁定窄带阈值：4.2 kHz 以上高频能量比例低于 `7.893876957903083e-05` 才应用恢复器。

## 5. 原 105 条全量恢复：失败结果

- overall ECAPA delta `-0.00288`，95% CI `[-0.02088,+0.01170]`；UTMOS `-0.03372`。
- telephone `+0.04977`，但 babble `-0.04848`、real noise `-0.03912`。

全量恢复路线失败。路由是在看过这批结果后形成，因此在这批数据上只能算 post-hoc。

## 6. 新随机种子确认集

固定模型、alpha、cutoff 与阈值后，使用 seed 9026 生成新的 105 条退化实例。它与原测试 sample_id 重叠 0、sample+seed 重叠 0，但共享同一组 40 名 held-out speakers；prompt/reference utterance ID 分别重叠 17/19。

| 指标 | baseline | routed V4 | delta |
|---|---:|---:|---:|
| ECAPA cosine | 0.61677 | 0.62233 | **+0.00556** |
| UTMOS | 2.63001 | 2.65358 | **+0.02357** |

ECAPA 95% CI `[+0.00174,+0.00998]`，paired Wilcoxon `p=0.01759`。触发 15/105：14 条 telephone、1 条 Opus；其余 90 条严格旁路。

Telephone 子集 ECAPA mean delta `+0.03405`，95% CI `[+0.01090,+0.05719]`，11/15 提升。路由漏掉 1 条 telephone；1 条极低高频能量 Opus 被触发且取得 `+0.07346`。

## 7. 当前判定与证据

确认集给出正向复现，但仍是单语料、共享 held-out speaker pool 的阶段性证据。下一步应做跨语料/语言/设备外部验证、人工 ABX/MOS 和更稳健的窄带检测校准。

- 本地：`artifacts/autodl-v4-2026-09-03/`
- 归档：`prompt_feature_v4_results_20260903.tar.gz`
- SHA-256：`a26d837ffbe246841a8e1865cce29dfda684aab9f71ae4485700e89dcb6e4fda`
