# V3 实现说明与已堵住的坑

1. 原 `.gitignore` 的 `models/` 会递归命中 `src/models/`，导致 GitHub 克隆即失败；现在只忽略根目录 `/models/`。
2. V2 质量输入不是 token，而是连续 `[SNR/30, bandwidth/8000]`；V3 继续用连续条件，并在训练期加扰动模拟推理估计误差。
3. 原 BWE 只看 `telephone: bool`，阈值附近会跳变；V3 用 sigmoid 带宽 gate 在原 mel 与 BWE mel 间连续插值。
4. batch 16 只有 15 个 in-batch negatives；V3 加可随 checkpoint 恢复的 4,096 FIFO embedding queue。
5. 原 best 在 resume 后重置为无穷，续训第一轮会错误覆盖；V3 checkpoint 保存并恢复 best、optimizer、scheduler、queue。
6. 原压缩/电话只是低通和量化；V3 正式 backend 调 ffmpeg 做 G.711、MP3、Opus 实际 encode/decode，并把 codec/bitrate 写进逐样本元数据。
7. “white/pink/HVAC/cafe”不能由一个混杂噪声池诚实区分；V3 正式协议改成 MUSAN real-noise、三说话人 babble、music 三类可追踪来源。
8. 原评测 prompt=合成句、clean reference=同一条退化前音频；V3 强制异句、同 speaker 不同 utterance。
9. 原 21 条只报均值；V3 保存逐样本 CSV，输出 bootstrap CI、paired Wilcoxon 与 Holm correction。
10. 原模型选择按复合 validation loss；V3 top-3 loss 候选由独立 selection speakers 的端到端 ECAPA 再选择，final test speakers 从未参与训练或选择。
11. UTMOS 接入明确写成 SpeechMOS v1.2.0 的 UTMOS22-strong 重实现，不冒充官方 ensemble 或人类 MOS。
12. `--formal` 是结果防伪门：synthetic fallback、跳过 denoiser/UTMOS、截断样本任一出现都会终止。
13. 噪声/RIR 也按文件 SHA-256 固定切成 train/selection/test，防止只隔离 speaker 却复用同一退化素材。
