# V3 正式评测协议

## 研究问题

给定同一说话人的受损提示音频，鲁棒前端是否能在不修改 CosyVoice 主干的前提下，比原始提示和通用增强提示更稳定地保持说话人身份，同时不牺牲合成自然度？

## 固定设计

- 7 类退化，每类 15 条，共 105 条配对样本；严重度 0/1/2 平衡轮转。
- 每个样本使用同说话人的两个不同干净 utterance：一个退化成 prompt，另一个只做 ECAPA 身份参考。
- 合成文本与 prompt 转写不同，防止同句内容泄漏。
- 三个系统消费同一 prompt/text：raw CosyVoice、MetricGAN+ 后 CosyVoice、V3 robust condition。
- 噪声/RIR 来自带哈希 asset manifest；telephone/compression 由 ffmpeg 真 codec 往返产生。

## 指标与统计

主指标是合成音频与独立参考的 ECAPA cosine；感知质量用 SpeechMOS v1.2.0 的 UTMOS22-strong 重实现；BWE 保真度为 mel-domain LSD，不能写成一般 waveform LSD。报告均值差、中位数差、win rate、5,000 次 bootstrap 95% CI、双侧 Wilcoxon signed-rank p；7 个退化分桶做 Holm 校正。

禁止删除失败样本、复用同一 prompt 充当身份参考、用 prompt 原句合成、把 UTMOS 预测说成人工 MOS，或根据结果事后更换协议。
