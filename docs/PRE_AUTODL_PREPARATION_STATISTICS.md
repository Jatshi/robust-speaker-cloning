# 鲁棒说话人克隆 V3：AutoDL 开机前准备工作统计

> 统计日期：2026-08-25  
> 统计对象：`robust-speaker-cloning` 本地工作树  
> 当前阶段：本地准备完成，正式 AutoDL 实验尚未开始

## 1. 文件放置路径

### 1.1 项目主目录

```text
C:\Users\jat_s\WorkBuddy\2026-06-02-09-04-45\项目文档研究\repos\robust-speaker-cloning
```

所有本轮新增或修改的源码、测试、脚本和说明文档都在这个 Git 仓库中，没有散落到当前 Codex 工作区。

### 1.2 原始夯实规划

```text
C:\Users\jat_s\WorkBuddy\2026-06-02-09-04-45\项目夯实规划_鲁棒说话人克隆.md
```

该文件只作为需求与问题清单，本轮没有修改它。

### 1.3 本地 smoke 产物

```text
C:\Users\jat_s\WorkBuddy\2026-06-02-09-04-45\项目文档研究\repos\robust-speaker-cloning\outputs\smoke-fixture
C:\Users\jat_s\WorkBuddy\2026-06-02-09-04-45\项目文档研究\repos\robust-speaker-cloning\outputs\smoke-checkpoints
```

这些是可重新生成的本地验证产物，已被 `.gitignore` 排除，不会进入 GitHub：

| 目录 | 大小 | 内容 |
| --- | ---: | --- |
| `outputs/smoke-fixture` | 约 0.48 MiB | 微型 manifest、teacher cache、mel cache |
| `outputs/smoke-checkpoints` | 约 303.51 MiB | CPU dry-run 的 best/last/epoch checkpoint 与 history |
| 合计 | 约 303.99 MiB | 仅用于证明训练—验证—保存—恢复链路可执行 |

### 1.4 当前不存在的本地大文件目录

以下正式实验目录目前尚未创建或下载：

- `data/`：AISHELL-1、MUSAN、RIRS_NOISES 尚未下载到本机仓库。
- `models/`：CosyVoice2-0.5B 正式权重尚未下载到本机仓库。
- `CosyVoice/`：固定 revision 的正式依赖仓库尚未部署到项目目录。
- `outputs/evaluation-v3/`：105 条正式评测尚未执行。

它们将在 AutoDL 上由准备好的脚本生成，不应放进 Git。

## 2. 工作树统计

统计口径为 `git ls-files --cached --others --exclude-standard`，即纳入已跟踪文件和未跟踪但应进入版本控制的文件，排除 `.gitignore` 中的模型、数据、缓存和 smoke 输出。

| 项目 | 数量 |
| --- | ---: |
| 非忽略文件总数 | 62 |
| Python 文件 | 36 |
| Python 代码总行数 | 1,933 |
| `src/` Python 文件 | 30 |
| `src/` Python 行数 | 1,706 |
| 测试文件 | 5 |
| 测试代码行数 | 174 |
| 测试函数 | 13 |
| Markdown 文档 | 8 |
| Bash/PowerShell 运行脚本 | 7 |
| 脚本总行数 | 151 |

目录级文件数量：

| 目录/根文件组 | 文件数 | 主要用途 |
| --- | ---: | --- |
| `src/` | 30 | 模型、训练、退化、评测、统计和打包 |
| `scripts/` | 7 | 本地 smoke、AutoDL 部署、下载、训练和选择 |
| `docs/` | 7 | 运行手册、协议、数据卡、证据账本和统计 |
| `tests/` | 5 | 13 个自动化测试 |
| `results/` | 2 | 原 V2 的 21 样本 CSV/JSON 证据 |
| `.github/` | 1 | GitHub Actions CI |
| `.light/` | 1 | 项目阶段 passport |
| `assets/` | 1 | 原 V2 README GIF |
| 项目根部其他文件 | 8 | README、依赖、许可、锁文件等 |

## 3. 本轮代码改动统计

当前分支与原始 `main` 的基线信息：

```text
branch: main
base HEAD: 8cb2beb (Release robust speaker cloning V2)
```

| 改动类型 | 数量 | 状态 |
| --- | ---: | --- |
| 修改已有文件 | 9 | 尚未 commit |
| 新增文件 | 41 | 尚未 commit |
| 本轮涉及文件合计 | 50 | 尚未 push |

修改的已有文件主要是 `.gitignore`、README、Demo、缓存构建、数据加载、CosyVoice 接口、V2 评测兼容入口和训练脚本。新增文件覆盖核心模型、正式退化、可信评测、AutoDL 流水线、CI、测试和文档。

## 4. 功能模块统计

### 4.1 核心模型与损失

路径：`src/models/`

- `robust_speaker_encoder.py`：6 层质量条件 Transformer、mel U-Net、软 BWE gate。
- `losses.py`：InfoNCE、CampPlus 接口蒸馏、BWE L1/MSE 联合损失。
- Encoder 参数量：4,922,113。
- BWE 参数量：3,904,592。
- 可训练参数合计：8,826,705。

### 4.2 训练可靠性

- 训练期质量条件扰动，缓解 oracle quality 与推理估计值的不一致。
- 4,096 条 FIFO embedding memory bank，增加对比负样本。
- checkpoint 恢复 optimizer、scheduler、best validation 和 memory bank。
- train / selection / test speaker 三路隔离。
- top-3 validation-loss checkpoint 再由 selection speaker 的端到端 ECAPA 选择。

### 4.3 真实退化和数据

- MUSAN real-noise、三说话人 babble、music 三类可追踪素材。
- measured RIR 卷积混响。
- ffmpeg G.711 A-law/μ-law、MP3、Opus 真正 encode/decode。
- noise/RIR 文件通过 SHA-256 固定拆分 train / selection / test。
- 正式模式缺少真实素材时直接失败，不允许静默回退到合成退化。
- 提供多域 tabular manifest 转换与 speaker 域前缀防冲突工具。

### 4.4 正式评测

- 7 类退化 × 每类 15 条 = 105 条 final-test 样本。
- prompt 与身份参考使用同一 speaker 的不同 utterance。
- synthesis text 与 prompt text 不同。
- raw CosyVoice、MetricGAN+、robust condition 三组对照。
- ECAPA cosine、mel-domain LSD、UTMOS22-strong 重实现分数。
- paired Wilcoxon、Holm 多重检验校正、5,000 次 bootstrap 95% CI。
- 逐样本 CSV 原子写入和中断恢复。
- `--formal` 禁止 synthetic fallback、跳过 denoiser/UTMOS 或截断样本。

### 4.5 工程化与交付

- 固定 CosyVoice、CosyVoice2-0.5B、SpeechMOS revision/model ID。
- AutoDL preflight 检查 CUDA、显存、磁盘、ffmpeg codec、模型和正式数据。
- aria2 16 连接断点下载。
- screen 后台运行和分阶段 `.done-*` 标记。
- formal evidence 打包时生成 SHA-256 manifest。
- GitHub Actions 自动执行 lint、compile 和 pytest。

## 5. 自动化验证统计

| 验证项 | 结果 |
| --- | --- |
| Pytest | 13 passed |
| Ruff | All checks passed |
| Python compileall | 通过 |
| Bash 脚本语法 | 5 个 AutoDL Bash 脚本通过 `bash -n` |
| CPU pipeline smoke | 训练、验证、best/last/epoch 保存和 reload 全部通过 |
| 参数 shape | Encoder `(B,80,T)→(B,192)`；BWE 保持 mel shape |
| CosyVoice 契约 | 文本规范化、分段、`llm_embedding`/`flow_embedding` 双注入通过 fake contract test |
| Git diff | `git diff --check` 通过 |
| 敏感信息 | 未发现本轮 SSH 密码或登录指令进入仓库 |

本地测试仍有两个非失败 warning：新版 torchaudio 的未来后端迁移提示，以及 Transformer `norm_first=True` 导致 nested-tensor 优化未启用。二者不影响当前正确性，AutoDL 固定依赖后还会再次核对。

## 6. 关键文档位置

| 文档 | 路径 |
| --- | --- |
| 项目总览 | `README.md` |
| AutoDL 正式运行手册 | `docs/AUTODL_RUNBOOK.md` |
| 开机前检查表 | `docs/PRE_AUTODL_CHECKLIST.md` |
| 正式评测协议 | `docs/EVALUATION_PROTOCOL.md` |
| 数据与许可记录模板 | `docs/DATA_CARD.md` |
| 声明—证据账本 | `docs/CLAIM_LEDGER.md` |
| 踩坑与实现说明 | `docs/IMPLEMENTATION_NOTES.md` |
| 本统计文档 | `docs/PRE_AUTODL_PREPARATION_STATISTICS.md` |

## 7. AutoDL 脚本位置

| 脚本 | 用途 |
| --- | --- |
| `scripts/bootstrap_autodl.sh` | 安装依赖、固定第三方版本、下载模型、运行 preflight |
| `scripts/download_public_data.sh` | 并发下载 AISHELL/MUSAN/RIR 并生成 asset manifest |
| `scripts/run_v3.sh` | manifest→teacher→cache→train→select→protocol→eval 主流水线 |
| `scripts/select_v3.sh` | top-3 checkpoint 端到端选择 |
| `scripts/launch_autodl.sh` | 在 screen 中启动并允许 SSH 断线 |
| `scripts/local_smoke.ps1` | Windows 本地 CPU 链路验证 |
| `scripts/run_v2.sh` | 保留的原 V2 复现入口 |

## 8. 当前完成边界

已经完成的是“正式实验开始前可以在本地完成的工程准备”。尚未完成、且必须在 AutoDL 上产生真实证据的部分是：

1. 下载正式模型、AISHELL-1、MUSAN 与 RIRS_NOISES。
2. 在 4090 上加载固定 CosyVoice revision 并跑 CUDA batch smoke。
3. 构建真实退化 feature cache 和 CampPlus teacher cache。
4. 完整训练 20 epoch。
5. 在 selection speakers 上选出 `selected.pt`。
6. 在 test speakers 上执行 105 条三组端到端合成与统计。
7. 根据真实结果更新 README、证据账本和 Hugging Face 权重。

当前没有 V3 性能数字，也没有把本地 smoke 当成正式实验。远端 AutoDL 尚未部署此项目；本轮文件也尚未 commit 或 push。
