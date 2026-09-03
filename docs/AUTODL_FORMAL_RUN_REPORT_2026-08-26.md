# Robust Speaker Cloning V3 AutoDL 正式运行报告

## 1. 结论先行

V3 已完成真实数据准备、教师缓存、正式特征缓存、20 epoch GPU 训练、独立选择集 checkpoint 选择，以及 105 条三路端到端正式评测。工程链路和可复现证据闭环成立，但当前 V3 的核心性能假设不成立：将自研鲁棒编码器输出作为 CosyVoice speaker embedding 后，说话人相似度显著低于直接使用退化提示音频的原始 CosyVoice 基线。

因此，当前可以公开“完成了真实退化、独立选择集、三路基线、断点恢复和统计检验的完整工程评测”，但不能公开“V3 提升了鲁棒说话人克隆性能”。负结果必须保留，下一版应先解决 speaker embedding 分布/接口对齐，而不是扩大训练轮数。

## 2. 运行环境与耗时

- 正式运行日期：2026-08-26
- 远端项目：`/root/autodl-tmp/robust-speaker-cloning`
- GPU：NVIDIA GeForce RTX 4080 SUPER，31.5 GiB 可见显存
- 正式评测耗时：1591 秒（26 分 31 秒）
- 正式评测 GPU 峰值：5166 MiB、128 W、47°C、SM 99%
- 正式评测产物：109 MiB；525 个逐样本 WAV
- 运行结束数据盘：130 GiB 总量，约 42 GiB 可用

## 3. 数据与训练事实

- AISHELL-1：400 个说话人压缩包全部验证并解压；141,925 个 WAV；抽样验证为 16 kHz。
- 正式子集：400 个说话人、6000 条语音；340/20/40 个说话人分别用于 train/selection/test。
- 真实退化资产：MUSAN 930 noise、426 speech、660 music；RIRS_NOISES 61,260 条 RIR；资产 manifest 含 SHA-256。
- 正式特征缓存：6000 条 clean、42,000 条退化样本，覆盖 7 类退化；后端明确为 `real_assets_and_ffmpeg`，无 synthetic fallback。
- CosyVoice 教师缓存：6000 个键、每个 192 维、全部有限。
- 正式训练：20/20 epoch、44,620 step、约 3676 秒；20 个 epoch checkpoint 与完整 history 均存在；最后 memory bank 为 4096×192。
- 验证损失前三名：epoch 18 = 2.36980、epoch 20 = 2.37970、epoch 17 = 2.47107。

## 4. 独立选择集 checkpoint 选择

三个候选均在未参与训练的 selection speakers 上完成 14 条端到端评测。选择标准是生成音频的独立 ECAPA 绝对均值，不使用 test speakers，也不按单条增量挑选。

| Epoch | Selection ECAPA 均值 | 结果 |
| ---: | ---: | --- |
| 17 | 0.40785 | 选中 |
| 20 | 0.40635 | 候选 |
| 18 | 0.39854 | 候选 |

三者差异很小，且三者相对原始基线均下降。epoch 17 只是“候选中最好”，不是“已经优于原始系统”。

## 5. 105 条正式三路评测

每个样本同时评测：原始 CosyVoice（退化提示）、自研鲁棒条件分支、MetricGAN+ 去噪后 CosyVoice。身份指标为独立 SpeechBrain ECAPA，相对音质指标为固定实现 `tarepan/SpeechMOS:v1.2.0:utmos22_strong`。

### 5.1 总体结果

| 系统 | ECAPA 相似度 | UTMOS |
| --- | ---: | ---: |
| 原始 CosyVoice 基线 | 0.60867 | 2.58748 |
| 自研鲁棒分支 | 0.39610 | 2.69787 |
| MetricGAN+ 去噪基线 | 0.52486 | 2.57975 |

鲁棒分支相对原始基线的 ECAPA 平均差为 −0.21257，中位数差为 −0.22233，95% percentile-bootstrap CI 为 [−0.23470, −0.18992]，胜率 4.76%，双侧 Wilcoxon p = 1.23×10⁻18。它小幅提高了平均 UTMOS，却以显著损失说话人身份相似度为代价。

### 5.2 分退化结果

| 退化 | 原始 ECAPA | 鲁棒 ECAPA | 去噪 ECAPA | 鲁棒−原始 | 鲁棒胜率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| real_noise | 0.64758 | 0.42220 | 0.52815 | −0.22538 | 0.0% |
| babble_noise | 0.61823 | 0.37072 | 0.51575 | −0.24751 | 0.0% |
| music_noise | 0.63554 | 0.36428 | 0.49174 | −0.27125 | 0.0% |
| reverb_noise | 0.64972 | 0.44299 | 0.52520 | −0.20673 | 0.0% |
| telephone | 0.50369 | 0.39024 | 0.47919 | −0.11345 | 20.0% |
| mp3 | 0.60674 | 0.38511 | 0.58619 | −0.22163 | 6.7% |
| opus | 0.59918 | 0.39711 | 0.54780 | −0.20206 | 6.7% |

七个分组的 95% CI 均不跨 0，经 Holm 校正后仍显著。telephone 的下降最小，说明窄带/电话条件值得作为后续修复的首个诊断子集，但不能据此宣称总体有效。

### 5.3 BWE 诊断

- 退化 mel-LSD 均值：19.5667 dB
- BWE 输出 mel-LSD 均值：7.5057 dB
- BWE gate：最小 0.000184、均值 0.824865、最大 0.999971

BWE 在声学重建指标上大幅降低 mel-LSD，但端到端身份相似度仍下降。这说明“频谱更接近干净语音”不足以保证“生成模型使用的身份条件更正确”。当前证据更支持 speaker embedding 分布/语义接口错配，或训练目标与端到端 TTS 身份目标不一致；这仍是诊断假设，需要 3.1 消融实验验证。

## 6. 完整性审计

以下检查全部通过：

- evaluation CSV 恰有 105 行、105 个唯一 sample ID；
- 7 类退化各 15 条；
- 9 个核心数值字段全部有限且无空值；
- 每条样本 5 个审计 WAV，共 525 个；
- prompt/reference 音频 ID 不同，prompt/synthesis 文本不同；
- summary 标记 `formal=true`、`three_way_baseline=true`；
- UTMOS 实现固定且记录到 summary；
- 退化元数据没有 synthetic/fallback；
- 回归测试 13/13 通过。

证据包：`outputs/robust-speaker-cloning-v3-evidence.tar.gz`。最终包的 SHA-256 写在同目录的 `robust-speaker-cloning-v3-evidence.sha256`；checksum 文件位于压缩包外，避免让包内文档引用包自身哈希而产生循环。包内 manifest 对实际源码、脚本、文档、配置和核心结果逐文件哈希。远端部署目录不含 `.git`，因此 manifest 如实标记 Git revision 不可用，而不是伪造提交号。

## 7. 本次踩坑与解决

1. ModelScope 模型仓库不提供可用 Git commit：改为固定 `master` 快照，并生成模型文件清单与 SHA-256。
2. CosyVoice GitHub 克隆出现 HTTP/2 失败：使用可重试下载与严格目录验证，避免“目录存在但源码不完整”。
3. Matcha 导入缺少 Lightning：固定 `lightning==2.2.4`，再执行真实 CosyVoice CUDA smoke。
4. SpeechBrain 1.1.0 与 Torch 2.3.1 的 AMP API 不兼容：固定 `speechbrain==0.5.16`，并先做 ECAPA CUDA smoke。
5. 候选循环的评测子进程继承了管道 stdin，第三方依赖读取 stdin 后吞掉 epoch 20：给评测进程加 `</dev/null`，复用 epoch 17/18 的原子 CSV，只补跑 epoch 20。
6. AutoDL 无法访问 GitHub Release 的 UTMOS 权重：从 Hugging Face 镜像下载同一 411,179,119 字节 checkpoint，并按公开 LFS SHA-256 `38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff` 校验。
7. 证据打包器强制要求 `.git`，远端部署快照因此失败：改为有 Git 时记录 revision/dirty，无 Git 时明确标记不可用，并对实际源码树逐文件哈希。

## 8. 下一版优先修复顺序

1. 对比教师 embedding、鲁棒 embedding、CosyVoice 原生 prompt embedding 的范数、协方差、余弦分布和主成分谱，确认接口错配位置。
2. 增加原生 speaker embedding 的分布对齐损失与端到端生成身份损失；先在固定小集上验证，再训练全量。
3. 做 `原始提示 / MetricGAN+ / 仅 BWE / 仅 encoder / BWE+encoder / oracle clean prompt` 六路消融，定位损失来自频谱修复还是条件编码。
4. 将 selection criterion 改为同时约束 ECAPA 身份、UTMOS 与最差分组表现，禁止以音质提升掩盖身份下降。
