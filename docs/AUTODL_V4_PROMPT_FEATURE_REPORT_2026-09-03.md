# V4 Prompt Feature Restorer：AutoDL 完整复盘

## 最终结论

原“从零蒸馏一个比 CAMPPlus 更强的 speaker encoder”路线不成立。最终方案完全保留官方 CAMPPlus、speech token 与 CosyVoice 主干，只在输入被判断为窄带时恢复 Flow 使用的 80 维 acoustic prompt feature；非窄带输入严格旁路。

新随机种子 105 条确认集上，ECAPA cosine 平均 `+0.00556`，95% CI `[+0.00174,+0.00998]`，Wilcoxon `p=0.01759`；UTMOS 平均 `+0.02357`。这是窄带专用、带失败保护的局部改进，不是全噪声通用增强。

## 为什么原方案不符合实际

CAMPPlus 是在大规模说话人数据上训练的成熟编码器。一个只在有限 AISHELL clean/degraded pairs 上从零训练的小网络，即使能拟合 teacher embedding 或频谱，也不会凭空得到更多身份判别信息。V3 的 105 条正式测试 ECAPA `-0.21257`。继续加 epoch 或扩大学生网络只会更充分地拟合错误代理目标。

曾怀疑 L2 normalization 是接口错误。真正运行 raw/unitnorm 回注并复核 CosyVoice 源码后发现，下游本来会归一化 embedding，两种回注输出完全一致，这条假设也被证伪。

## 如何找到真正瓶颈

组件级 clean Oracle 一次只替换一个条件，其他输入保持退化，最终通过完整 CosyVoice 合成并用独立 reference 的 ECAPA 打分。clean embedding only 几乎没有稳定 headroom；clean token only 下降；clean `prompt_speech_feat` 7/7 改善，mean `+0.03534`；full clean Oracle mean `+0.04104`。因此训练目标从 speaker embedding 转到 Flow 的 prompt feature。

## 模型在做什么

CosyVoice 将 24 kHz prompt 音频变成 80×T 的 mel feature。恢复器把标准化 feature 与两个质量条件拼接，经过 8 个轻量膨胀时序残差块预测残差。输出层零初始化：

`feature_out = feature_in + alpha × gate(quality) × residual`

`alpha=0` 时数值上就是官方输入；因此检测不确定或非目标输入可直接旁路。模型只有 161,361 个可训练参数，成熟 TTS 组件全部冻结。

## 数据链路

AISHELL-1 中 2,000 条 clean prompts 经过 7 类退化形成 14,000 个配对样本。真实噪声来自 MUSAN，混响使用 measured RIR，MP3/Opus 走真实 ffmpeg round trip，telephone 做窄带链路。训练缓存的是 CosyVoice 精确前端输出。

训练/验证以 speaker 隔离。validation feature L1 从 `2.28309` 降至 `0.47711`，但这只说明特征拟合成功，不代表端到端收益。

## 关键坑与解决方式

1. **代理指标陷阱**：V3 的 mel-LSD 改善与最终 ECAPA delta 的 Spearman ρ≈`-0.009`。解决：所有选择最终回到端到端 TTS。
2. **强教师替换陷阱**：小数据从零网络无法合理超越 CAMPPlus。解决：成熟组件冻结，只修复 Oracle 证明有 headroom 的分支。
3. **静态代码猜因果**：看到范数不同就归因尺度。解决：做波形级 reinjection 对照并追到下游 `F.normalize`。
4. **离线 loss 漂亮但线上无效**：feature L1 大幅下降，但全量端到端总体仍为负。解决：保留负结果，按退化拆分。
5. **post-hoc 污染**：在原 test 看到 telephone 获益后才设计路由。解决：阈值只用 selection 锁定，再跑 seed 9026 新实例确认集。
6. **安全部署问题**：模型会伤害 babble/real noise。解决：输入高频能量 pre-router；不触发时 `effective_alpha=0`，90/105 条严格零改动。
7. **路由并不完美**：确认集中漏 1 条 telephone、触发 1 条 Opus。解决：如实报告 confusion；下一版加跨语料校准和 abstain 灰区。

## 面试中怎样准确描述

> 我最初尝试蒸馏退化语音到 CAMPPlus 条件空间，但 105 条端到端实验显著下降。我没有继续调参，而是用组件 Oracle 拆解 CosyVoice conditioning，发现主要 headroom 在 Flow 的 acoustic prompt feature。随后训练了 161k 参数、零初始化的残差恢复器；全量应用仍会伤害噪声条件，所以我用 selection 锁定高频能量路由，并在新随机种子 105 条确认集上得到 ECAPA +0.00556、95% CI 不跨 0，非目标输入严格旁路。

不要说“全面超过 CAMPPlus”或“所有噪声都提升”。真正有价值的工程能力是：证伪假设、组件诊断、保留失败实验、设计 no-op fallback，并用未参与定规则的新实例验证。

## 后续最值得做的实验

1. 用 CN-Celeb/VoxCeleb 或真实用户录音做外部说话人与设备验证。
2. 将单阈值替换为校准窄带 detector，并加入灰区拒绝恢复。
3. 增加人工 speaker-similarity ABX 与自然度 MOS。
4. 对 codec、采样率、EQ 与带宽边界做 reliability curve。
