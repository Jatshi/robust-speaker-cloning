# Robust Speaker Cloning V4 实验追踪

| 阶段 | 状态 | 产物/判据 |
|---|---|---|
| V3 失败定量复盘 | 已完成 | 105 条正式 CSV；custom vs raw mean delta = -0.21257 |
| 官方接口核对 | 已完成 | 官方 CAMPPlus 输出不做 L2 normalize；V3 两处错误 normalize 已定位 |
| V4 方案设计 | 已完成 | Frozen CAMPPlus residual calibration + alpha=0 fallback |
| 本地代码实现 | 已完成 | V4 独立文件，不覆盖 V3 审计链路；缓存、训练、诊断、选择、正式评测均已实现 |
| 本地单元/冒烟测试 | 已完成 | 2026-09-03：19 passed；Ruff 与 compileall 通过 |
| AutoDL Gate A | 待执行 | baseline/raw/unitnorm/clean oracle，7 条 |
| CAMPPlus 缓存 | Gate A 后 | 仅 Gate A 证明有 headroom 后运行 |
| Adapter 训练 | Gate A 后 | selection embedding 指标与 best checkpoint |
| Alpha 端到端选择 | 训练后 | 14 条，alpha 含 0 |
| 正式 105 条测试 | Gate C 后 | 仅 selection 正向才运行 |
| 证据归档 | 待执行 | F 盘 + 报告 + claim evidence table |

## 当前阻塞

本地准备已完成。已知的 38651、39405、35699 端口均无法建立 TCP 连接；需要本次开机对应的最新 SSH 地址和密码。
