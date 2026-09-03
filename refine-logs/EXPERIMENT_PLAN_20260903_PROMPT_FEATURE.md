# Robust Speaker Cloning V4：Prompt Feature 安全恢复实验计划

日期：2026-09-03  
状态：已执行，结果见 `EXPERIMENT_TRACKER_20260903_PROMPT_FEATURE.md`

## 1. 问题重定义

V3 的核心错误不是“训练不够久”，而是试图用有限配对数据从零训练一个网络去替代成熟的 CAMPPlus。若学生模型只看到小规模 AISHELL 配对和有限退化，它缺少超越强教师的身份判别信息；频谱重建损失也不等同于最终克隆身份保持。因此 V4 不再替换成熟组件，而是先用端到端 Oracle 定位退化到底损坏了 CosyVoice 的哪个条件分支。

目标不是宣称对所有噪声普遍提升，而是在可观测、可路由、不会伤害非目标输入的条件下，提高窄带参考音频的克隆身份一致性。

## 2. 必须先回答的诊断问题

1. 官方 CAMPPlus 回注是否能精确复现 baseline？
2. 单位归一化是否真的改变 CosyVoice 输出？
3. 将 embedding、speech token、prompt feature 分别换成 clean Oracle，哪个分支存在最大端到端 headroom？
4. 如果学习 feature restoration，离线特征误差下降能否转化为最终 ECAPA/UTMOS 增益？
5. 若仅部分退化获益，能否用不依赖标签的输入统计量在推理前安全识别？

## 3. 诊断门禁

固定同一随机种子与同一 synthesis text，对 7 类退化各取 1 条，比较：official degraded baseline、raw/unit-norm CAMPPlus reinjection、clean embedding/token/prompt feature 单组件替换、clean token+feature 与 full clean Oracle。

只有某个组件的 clean Oracle 在最终 ECAPA 上显示明确 headroom，才允许训练该组件的恢复器。Oracle 仅用于定位，不进入训练或正式推理。

## 4. 最终模型

输入是 CosyVoice 官方 Matcha 前端产生的 `[B,80,T]` prompt feature，以及两个归一化质量统计量。网络包含 82→128 输入投影、8 个深度可分离时序残差块、128→80 输出头和质量 gate。输出层零初始化，保证初始状态与 `alpha=0` 都是严格 no-op。

总可训练参数 161,361。CAMPPlus、speech tokenizer、CosyVoice LLM、Flow 与 vocoder 全部冻结。

## 5. 数据与切分

- AISHELL-1 中 400 名训练/验证说话人，每人 5 条，共 2,000 条 clean prompt。
- 每条生成 7 种真实或实际编解码退化，共 14,000 对 exact prompt features。
- speaker-disjoint：11,900 个训练 pair，700 个 validation pair；正式测试使用独立的 40 名 held-out speakers。
- 退化：babble、music、real noise、measured RIR、MP3、Opus、telephone narrowband。

缓存使用 CosyVoice 实际参数：24 kHz、n_fft=1920、80 mel、hop=480、win=1920、fmax=8 kHz、`center=False`，而非另造一套 log-Mel。

## 6. 优化目标

`L = L_smoothL1(z_pred,z_clean) + 0.25 L_MSE + 0.20 L_temporal + 0.10 L_mel_axis + 0.25 L_clean_identity + 0.01 L_trust_region`

坐标损失恢复 clean feature；temporal/mel-axis 项约束局部结构；clean identity 让 clean 输入保持不变；trust region 限制无必要的大幅改写。训练 25 epoch，按 validation total loss 保存最佳 checkpoint。

## 7. 选择、失败判据与安全路由

先在 selection 上搜索 `alpha∈{0,.25,.5,.75,1}`。如果全量应用的总体 CI 跨 0 或某些类别明显回归，不得包装成通用增强器。

窄带路由只允许使用 selection 数据确定阈值：计算 STFT 中 4.2 kHz 以上能量占 300–7800 Hz 总能量的比例。当比例低于锁定阈值时应用 `alpha=1`，否则 `effective_alpha=0` 严格旁路。阈值锁定后不得查看正式/确认集再修改。

## 8. 评测协议

- 7 类 × 15 条 = 105 条；prompt 与 identity reference 使用不同 utterance，synthesis text 也不同。
- baseline 与候选使用相同合成随机种子。
- 主指标 SpeechBrain ECAPA cosine；辅指标 UTMOS22 strong。
- paired two-sided Wilcoxon、percentile bootstrap 95% CI；分组结果 Holm 校正。
- 原 105 条若已用于发现路由，只能记为 post-hoc；必须再生成新随机种子确认集。

## 9. 结论边界

确认集总体 CI 下界大于 0 时，只能说“窄带路由方案在同一 held-out speaker pool 的新随机退化实例上复现了平均提升”。不得说对所有噪声均提升、超越 CAMPPlus、已证明跨语言/设备泛化或已有人工 MOS 结论。
