# 实验记录

## 状态

- 工程流水线：已完成
- 模拟数据端到端测试：已通过
- 真实比赛数据：已放入并通过字段校验
- WSL2 GPU 环境：CatBoost GPU 与 LightGBM CUDA 均已验证
- 首个真实 OOF：GPU 两折短实验已运行；正式固定五折基线尚未运行
- 首次 Kaggle 提交：已生成本地 GPU smoke submission，尚未上传 Kaggle

## 实验表

| 实验 | 特征 | 模型 | CV | OOF AUC | Public LB | 结论 |
|---|---|---|---|---:|---:|---|
| smoke-modern | 原始 + 人工特征 | CatBoost / LightGBM / ExtraTrees / Logistic | 3-fold demo | 仅验证工程 | — | 四类模型、融合和提交生成均已跑通 |
| real-data-readonly | 12 个真实字段 | 对抗验证 ExtraTrees | 100k + 100k / 3-fold | 0.544574 | — | train/test 存在轻微漂移，主要来自字段分布和缺失率差异 |
| real-sample-smoke | 原始 + 人工 + 缺失特征 | CatBoost / LightGBM | 100k 抽样、单次 80/20 | 0.951293 / 0.955492 | — | 真实缺失和类别路径已跑通；该分数仅作兼容性检查，不用于模型选择 |
| real-gpu-smoke | 原始 + 人工 + 缺失特征 | CatBoost GPU / LightGBM CUDA | 完整 691,369 行 / 2-fold | 0.940539 / 0.947684 | — | GPU 全链路通过；耗时 26.4s / 17.0s，本结果不可与固定 5-fold 正式实验直接比较 |

## GPU 验证记录

- 设备：NVIDIA GeForce RTX 5090 D，计算能力 12.0；
- CatBoost：1.2.10，`task_type=GPU`；
- LightGBM：4.7.0，CUDA 12.9 源码构建，`device_type=cuda`；
- GPU 自检：8,192 行、24 个数值特征，两种后端均成功拟合；
- 真实两折短实验：CatBoost 26.4 秒，LightGBM 17.0 秒；
- smoke 融合权重：LightGBM 1.0、CatBoost 0.0；
- smoke submission：`outputs/gpu_smoke/submission.csv`；
- 正式比较仍必须回到固定五折，不能把两折 smoke AUC 当作冲榜结论。

## 真实数据快照

- 训练集：691,369 行、12 个输入字段；
- 测试集：296,302 行；
- 正类比例：0.709424；
- `daily_screen_time_hours` 单变量排序 AUC：0.865394；
- `weekend_screen_time` 单变量排序 AUC：0.850397；
- `social_media_hours` 单变量排序 AUC：0.817737；
- 训练集完全重复特征行：0；
- 测试集与训练集精确匹配行：2；
- 所有输入字段均存在缺失，单个缺失指示本身几乎没有标签信号。

## 首轮真实数据实验计划

| 优先级 | 实验目录 | 改动 | 需要回答的问题 |
|---:|---|---|---|
| 1 | `outputs/audit_v1` | 完整审计 + 对抗验证 | 是否存在泄漏、重复行或 train/test 漂移？ |
| 2 | `outputs/all_features` | 保留全部字段 | 强基线 OOF 是多少？ |
| 3 | `outputs/no_weak_categories` | 删除三个弱类别字段 | 低信号类别是否只会增加噪声？ |
| 4 | `outputs/with_id` | 加入 `id` | ID 是否包含生成批次信号？ |
| 5 | `outputs/raw_features` | 关闭人工特征 | 行为比例与差值是否带来稳定提升？ |

## 记录规则

每个有效实验至少记录：

- 配置和输出目录；
- 模型五折 OOF AUC；
- 各折标准差；
- 与主模型的 OOF 排名相关性；
- 融合前后提升；
- Public LB；
- 是否在不同随机种子下复现；
- 最终保留或放弃的原因。
