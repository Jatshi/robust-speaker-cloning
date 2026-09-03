# Robust Speaker Cloning V4 实验计划

当前正式版本为 [EXPERIMENT_PLAN_20260903_PROMPT_FEATURE.md](./EXPERIMENT_PLAN_20260903_PROMPT_FEATURE.md)。

2026-09-03 的接口诊断推翻了旧版“CAMPPlus 数值尺度错误”假设：CosyVoice 下游会再次执行 `F.normalize`，raw 与 unit-norm 回注的波形完全相同；clean CAMPPlus Oracle 的端到端收益也很小。当前方案因此改为冻结 CAMPPlus、speech token 与 CosyVoice 主干，只恢复 Flow 真正消费的 80 维 `prompt_speech_feat`，并用选择集锁定的窄带检测器进行安全路由。
