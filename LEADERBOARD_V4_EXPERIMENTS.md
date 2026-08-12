# V4 冲击 Top 5%：新架构实验记录

更新日期：2026-08-12
工作分支：`main`（远端默认分支；仓库没有 `master`）

## 结论

本轮没有找到足以替换 V3 主候选的可信模型。当前主提交仍为：

```text
outputs/leaderboard_v3_combined/submission.csv
```

它的固定 5 折 OOF ROC-AUC 为 `0.969750802`，已记录的 Public LB 为
`0.97084`（2026-08-06 快照，Rank 42 / 约 770，约 Top 5.5%）。本轮最好的新结果是
GAM/EBM 四成员接入完整 90 列链路后的 `0.969755185`，仅提升
`+0.000004383`，低于预注册的 `+0.000020` 主候选替换门槛，且测试预测与 V3 的
Spearman 相关为 `0.999919`。这一级别更像微小残差修正，不足以支持“已进入 Top 5%”
的判断。

## 本轮固定验收规则

- 所有候选必须使用 `StratifiedKFold(5, shuffle=True, random_state=42)`。
- 不根据单一 pooled OOF 峰值采用：至少 4/5 折同向。
- 新一级模型先做代表折筛选；单模既要有足够强度，也要与现有最终预测低相关。
- 完整最终链路相对 V3 至少提升 `+0.000020` 才替换主提交。
- rank/band 权重保持 V3 固定值，不与新模型同时搜索，避免双重调参。

## 已执行实验

| 实验 | 结果 | 相对 V3 | 决策 |
|---|---:|---:|---|
| Missingness regime V2 / count-family gating | 代表折混合、部分折下降 | 未达门槛 | 淘汰 |
| 缺失数 specialist + 桶内重排 | `0.969751967` | `+0.000001165` | 淘汰 |
| 4 个 GAM/EBM 加入 86 列完整栈 | `0.969755185` | `+0.000004383`，4/5 正 | 保留研究产物，不替换 |
| 新 FM：同步缺失增强 + cross + pairwise rank loss | fold 0 `0.966501901` | 与旧 FM Spearman `0.991470`；1% 融合仅约 `+0.0000005` | 两折前止损 |
| 浅层 XGBoost 非线性 meta | 折 0/2 pooled `0.969350301` | 与当前最终 rank 融合最大约 `+0.0000014` | 淘汰 |
| 约束几何低秩 DCNv2 | 折 0/1 pooled `0.943712032` | 相关降至 `0.952433`，但从 0.5% 权重起两折均下降 | 淘汰 |

上表中的 FM、非线性 meta 和 DCNv2 是代表折快速筛选，不是完整 5 折结论；它们没有达到
预先设定的晋级条件，因此没有继续消耗算力跑全 5 折。GAM/EBM 是本轮唯一完成全部 5 折、
完整二层融合与两段后处理的新增候选。

GAM/EBM 完整复测的阶段结果：

| 阶段 | 86 列 AUC | 90 列 AUC | 增量 | 正向折 |
|---|---:|---:|---:|---:|
| Global | 0.969708243 | 0.969714079 | +0.000005836 | 4/5 |
| Regime | 0.969733184 | 0.969737242 | +0.000004058 | 5/5 |
| Rank mix | 0.969741844 | 0.969746308 | +0.000004464 | 5/5 |
| Band 1 | 0.969743709 | 0.969748033 | +0.000004324 | 5/5 |
| 最终 Band 2 | 0.969750802 | 0.969755185 | +0.000004383 | 4/5 |

对应报告位于：

```text
outputs/additive90_full_eval/full_eval_report.json
outputs/additive90_full_eval/stage_summary.csv
outputs/additive90_full_eval/per_fold_deltas.csv
```

同时生成了一个仅供 Kaggle A/B 验证的备用文件：

```text
outputs/additive90_full_eval/submission_additive90_challenger.csv
```

该文件已通过列名、行数、ID 顺序、有限值和概率范围检查；它不是新的主提交，原因是
OOF 增益没有达到替换门槛。当前主提交仍是 V3。

## 新增的可复现实验代码

### 1. GPU FM-rank trainer

`scripts/train_fm_rank.py` 修复了公开 FM 脚本不能直接运行及缺失增强不一致的问题：

- 使用项目配置和原始数据路径，不依赖缺失的 `common.py`；
- CUDA 不可用时直接失败，不静默回退 CPU；
- exact lookup、coarse lookup、PLR 和依赖派生特征同步隐藏；
- BCE 与 pairwise logistic ranking loss 联合优化；
- 生成 `folds.csv`、OOF/test、raw-logit parquet 和完整 `metrics.json`。

正式 OOF 默认使用预先固定的 epoch 和最终 EMA 权重，不读取外层验证折来挑 checkpoint。
`--screen-early-stop` 只允许在方向筛选时显式开启；其分数带有 checkpoint 选择偏差，不能作为
正式 OOF 证据。本轮 FM 与 DCNv2 的代表折数字按筛选口径记录，且两者均大幅低于晋级线，
因此该偏差不会改变淘汰结论。

### 2. 非线性 meta 筛选器

`scripts/screen_nonlinear_meta.py` 在 frozen OOF 上试验浅层强正则 CatBoost/XGBoost
二层模型。它只用于筛选，不接入生产链路；当前实验证明树形二层的收益不足。
默认采用固定迭代次数；只有显式传入 `--screen-early-stop` 才使用外层验证折 early stopping。

### 3. Constraint-geometry DCNv2

`scripts/train_budget_cross.py` 故意放弃精确值查表，只使用：

- 连续字段 rank-Gauss；
- 逐字段缺失 mask、缺失 pattern 与缺失数量；
- `daily >= social + gaming + work/study` 约束产生的上下界、宽度和有效性；
- 屏幕时间行内统计与比率；
- 两层低秩 cross network、MLP、BCE + pairwise ranking loss。

随机缺失增强先作用于原始字段，再在线重算全部约束特征，防止从派生字段旁路恢复
被隐藏的值。该视图虽然明显低相关，但强度不足，证实“低相关”必须与“足够准确”同时成立。

## GPU 环境

本轮在 WSL Ubuntu 22.04 / NVIDIA GeForce RTX 5090 D 上安装并验证：

```text
torch 2.11.0+cu128
CUDA runtime 12.8
```

`requirements-gpu.txt` 固定了同一安装来源。实际 GPU 矩阵运算和两个神经网络训练均已
通过，训练日志中的设备为 `cuda:0`，不是 CPU 回退。

## 下一步判断

当前 86 列栈对相似模型和二层模型已高度饱和。继续调 Logistic `C`、band 权重、增加
GBDT seed 或加入弱而低相关的模型，可信空间约只有 `1e-6` 到 `1e-5`。若仍要显著推进，
需要新的强一级模型达到接近 `0.968+` 单模 AUC，并具备不同误差结构；同时应为最终候选
增加 repeated/nested 验证，避免把第五位小数的 OOF 噪声当作提升。
