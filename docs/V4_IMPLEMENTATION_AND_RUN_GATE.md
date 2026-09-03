# V4 实现与运行门禁（2026-09-03 修订）

旧版冻结 CAMPPlus residual adapter 方案已被接口与 Oracle 诊断否决，不再作为正式 V4 路线。CosyVoice 下游本身会归一化 embedding，且 clean embedding only 的端到端 headroom 很小。

当前 V4 仅恢复 CosyVoice Flow 使用的 80 维 `prompt_speech_feat`：

```bash
bash scripts/run_prompt_feature_v4.sh cache
bash scripts/run_prompt_feature_v4.sh train
bash scripts/run_prompt_feature_v4.sh select
python -m src.select_prompt_feature_router \
  --evaluation outputs/prompt-feature-v4/selection/evaluation.csv \
  --audio-root outputs/prompt-feature-v4/selection \
  --output outputs/prompt-feature-v4/selection/router.json
```

只有 `router.json` 的 `gate_passed=true` 才可锁定阈值。正式评测必须传入该阈值；非触发输入的 `effective_alpha` 必须为 0，并在逐样本 CSV 中记录。

无条件 alpha=1 的原 105 条总体为负，因此不得作为部署方式。由于路由是在查看原结果后形成的，必须用新随机种子协议确认；当前确认集结果见 [AUTODL_V4_PROMPT_FEATURE_REPORT_2026-09-03.md](./AUTODL_V4_PROMPT_FEATURE_REPORT_2026-09-03.md)。
