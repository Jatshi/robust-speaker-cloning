# AutoDL 正式运行手册

## 1. 当前边界

本地准备已经覆盖源码、测试、CPU 训练 dry-run、真实退化入口、正式评测协议和断点脚本。尚未发生的只有必须依赖 AutoDL GPU/大数据/大模型的事实：真实数据缓存、正式训练、105 条三组 CosyVoice 合成与指标计算。因此在这些阶段结束前，只能说“链路准备完毕”，不能说“V3 已提升”。

建议资源：RTX 4090 24 GB、系统内存 64 GB 以上、可用数据盘至少 80 GiB。现有 200 GB 数据盘足够；训练阶段主要显存来自 CosyVoice teacher/evaluation 与前端模型，前端本身不足 9M 参数。

## 2. 首次部署

```bash
cd /root/autodl-tmp
git clone https://github.com/Jatshi/robust-speaker-cloning.git
cd robust-speaker-cloning
export PROJECT_DIR=$PWD
bash scripts/bootstrap_autodl.sh
```

`bootstrap_autodl.sh` 会安装依赖、固定 CosyVoice revision、下载模型，并执行单元测试和 prepare preflight。若 CosyVoice 安装修改了 Torch/CUDA，先检查 Torch 与 torchaudio 是否同版本系、CUDA 是否可用，再开始付费训练。

## 3. 公共数据

```bash
export PROJECT_DIR=/root/autodl-tmp/robust-speaker-cloning
bash scripts/download_public_data.sh
```

下载采用 aria2 16 连接和断点续传。脚本准备 AISHELL-1、MUSAN 与 RIRS_NOISES，并生成带 SHA-256 的 `data/degradation_assets.json`。下载后在 `docs/DATA_CARD.md` 填真实下载日期、来源版本与许可核对结果。

## 4. 分阶段执行

```bash
bash scripts/run_v3.sh manifest
bash scripts/run_v3.sh teacher
bash scripts/run_v3.sh cache
python -m src.preflight --stage train --project-root "$PROJECT_DIR"
bash scripts/launch_autodl.sh train
```

用 `screen -r robust_clone_v3`、`tail -f outputs/logs/train.log` 和 `nvidia-smi` 观察。训练每个 epoch 都写 `last.pt`、`best.pt`、`epoch-NNN.pt`。重启后再次启动 train 会恢复优化器、学习率、best 值和 memory bank。

训练结束后：

```bash
bash scripts/run_v3.sh select
bash scripts/run_v3.sh protocol
bash scripts/launch_autodl.sh eval
```

评测逐样本原子更新 CSV，意外中断后 `--resume` 跳过已完成 ID。

## 5. 完成判据

- `history.json` 有 20 个 epoch，且不是 CPU smoke；
- feature-cache metadata 的 backend 为 `real_assets_and_ffmpeg`；
- eval protocol 恰有 105 行，prompt/reference ID 不同；
- evaluation CSV 恰有 105 个唯一 ID；
- `selection.json` 表明 top-3 loss 候选已由独立 selection speaker 的端到端 ECAPA 选择；
- 每行三组相似度与 UTMOS 非空；
- summary 包含 overall、7 个退化分组、CI、Wilcoxon p、Holm p；
- 负结果保留，不挑样本、不删退化类型。

## 6. 失败恢复

- 下载失败：重跑下载脚本，aria2 自动续传。
- teacher/cache 失败：修复后删除对应 `outputs/.done-*` 标记再重跑。
- CUDA OOM：batch 16 调到 8并记录，不减少数据/评测样本。
- checkpoint 损坏：从最近 epoch checkpoint 恢复，保留损坏文件追因。
- ffmpeg 缺 codec：preflight 会失败，先补全 encoder。
- UTMOS 下载失败：单独预热 scorer，不用空值继续正式评测。
