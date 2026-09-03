# 数据卡（正式运行前填写）

| 用途 | 数据 | 代码中的角色 | 是否进 Git |
| --- | --- | --- | --- |
| 干净多说话人训练 | AISHELL-1 | teacher、clean/degraded prompt | 否 |
| 真实噪声 | MUSAN（可追加 DNS 子集） | 噪声混合 | 否 |
| 真实脉冲响应 | RIRS_NOISES | 卷积混响 | 否 |
| codec | 本机 ffmpeg | G.711、MP3、Opus 往返 | 仅版本信息 |

正式运行需填写：下载日期、各数据许可核对、ffmpeg 完整版本、speaker/utterance 数、split 交集检查、asset manifest SHA-256、所有过滤规则。

`prepare_degradation_assets` 记录每个 noise/RIR 文件的 SHA-256，并按哈希固定划分 train/selection/test，避免正式测试复用训练期同一噪声或 RIR；但这不能替代许可核对。AISHELL-3/4、WenetSpeech/GigaSpeech 属于后续泛化实验；加入前需域标签、speaker-disjoint split 与许可记录，不能把“支持导入”写成“已验证”。

后续语料先整理成 `speaker,audio_id,wav_path,text` CSV，再用 `python -m src.manifest_tools convert` 转换；多个域用 `merge` 合并。工具会给 speaker 加域前缀，避免不同数据集重名。该入口不代表仓库已经下载或训练这些语料。
