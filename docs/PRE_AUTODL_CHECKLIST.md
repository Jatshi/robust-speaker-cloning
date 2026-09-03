# AutoDL 开机前检查表

- [x] `.gitignore` 只忽略根目录 `/models/`。
- [x] encoder、BWE、loss 源码恢复，参数量实测。
- [x] 10 个单元测试覆盖模型、损失、退化、路由、质量、统计、协议。
- [x] CPU 小缓存训练/验证/checkpoint reload/memory bank dry-run 完成。
- [x] CosyVoice 与 SpeechMOS revision 固定。
- [x] AutoDL bootstrap、并发下载、分阶段、screen、日志、断点脚本完成。
- [x] 105 条独立参考 + 异句协议、三组评测、增量 CSV 完成。
- [x] 正式模式禁止 fallback、skip baseline/UTMOS、max-samples。
- [ ] AutoDL 验证固定 CosyVoice commit 的 condition injection 接口。
- [ ] 下载并哈希真实数据，填写 data card。
- [ ] 4090 跑 1 batch CUDA smoke，确认显存/吞吐/AMP 后再正式 20 epoch。

最后三项必须在 AutoDL 开机后完成，不能由本地 CPU 伪造。
