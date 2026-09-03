# Claim–Evidence Table

| 编号 | 暂定主张 | 当前证据 | 状态/限制 |
|---|---|---|---|
| C1 | V3 自训练替代编码器降低了最终说话人一致性 | 105 条正式测试：mean ECAPA delta -0.21257，95% CI [-0.23470, -0.18992]，win rate 4.76% | 已支持负向结论 |
| C2 | V3 的 BWE 频谱改善不能转化为身份改善 | 105 条上 mel-LSD improvement 与 ECAPA delta 的 Spearman ρ≈-0.00937，p≈0.9244 | 已支持“无观察到关联”，不是因果证明 |
| C3 | V3 失败的主因是 CAMPPlus 单位归一化 | raw/unit-norm 回注相对官方 baseline 的波形 MAE 都为 0；CosyVoice Flow/LLM 内部会 `F.normalize` | **已证伪**；不得继续作为失败解释 |
| C4 | CAMPPlus 是当前主要瓶颈 | 7 条组件 Oracle：clean embedding only +0.00275，CI 跨 0；clean feature only +0.03534，7/7 改善 | **已证伪**；主要可利用 headroom 位于 `prompt_speech_feat` |
| C5 | 全退化无条件恢复 prompt feature 能普遍提升 | 原始 105 条：总体 -0.00288，CI [-0.02088, 0.01170]；babble、real noise 明显拖累 | **已证伪**；完整保留负结果 |
| C6 | 选择集锁定的窄带安全路由可避免非目标退化回归 | 新随机种子 105 条：90 条严格旁路；15 条触发（14 telephone、1 Opus） | 已支持路由行为；确认集与原测试共享 40 名 held-out speakers |
| C7 | 窄带路由 + prompt feature restorer 在确认集提升说话人相似度 | 新 105 条：mean ECAPA delta +0.00556，95% CI [+0.00174,+0.00998]，Wilcoxon p=0.01759；UTMOS +0.02357 | 初步支持；仍需跨语料、跨语言和人工 MOS 验证 |
| C8 | 对 telephone 子集改善 | 新 15 条：ECAPA +0.03405，CI [+0.01090,+0.05719]，11/15 胜；原测试 15 条 +0.04977 | 两轮一致为正；确认集 Holm 校正后的分组 p=0.1102，不宜声称“所有电话样本均提升” |

所有未来简历、README 和面试表述必须服从本表；Oracle、smoke、selection 和被检查后才设计的 post-hoc 路由结果不能冒充独立泛化结论。推荐表述是“在同一 held-out speaker pool 的新随机退化确认集上得到初步复现”，不是“普遍解决所有噪声下的说话人克隆”。
