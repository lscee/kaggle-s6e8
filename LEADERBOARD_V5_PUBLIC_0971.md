# Playground S6E8：0.971+ 公开方案审计与 V5 实验

更新日期：2026-08-17

## 结论

最新公开榜已经进入高度拥挤区：榜首约 `0.97132`，前 49 名均不低于
`0.97113`。但公开出现的 `0.971+` 并不都代表新的可泛化模型：部分结果来自
公开提交互相 rank blend，部分来自针对 Public LB 的微排序探测。

本轮只吸收了能够解释、复现并用 OOF 验证的两类信息：

1. 合成数据生成器留下的首位小数、有效小数长度、精确数值层级和屏幕时长残差；
2. Contextualized Deep Univariate Spline Transformer 带来的低相关非线性表示。

最终得到三个可提交候选：

- 稳健同折候选：`outputs/leaderboard_v4_artifact_stack/submission.csv`，OOF
  `0.969774678`；
- seed-42 样条进入 90 列完整 stack：
  `outputs/leaderboard_v5_spline_seed42_stack/submission.csv`，OOF `0.969781998`；
- 双 seed 样条挑战候选：
  `outputs/leaderboard_v5_dual_spline_rank_w04/submission.csv`，OOF `0.969788823`。

最强候选相对原 V3 的 `0.969750802` 累计提升 `+0.000038021`，但仍不能据此
承诺 Public LB 一定超过 `0.971`。

## 对公开 0.971+ 的核验

### 值得参考：非线性样条模型

来源：

- [模型讨论](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735421)
- [公开 Notebook](https://www.kaggle.com/code/ern711/contextualized-deep-univariate-spline-transformer)

核心不是“再训练一个 Transformer”，而是先把每个数值字段建模成多分辨率的一维
可学习样条，再通过很浅的 attention 混合字段。公开 Notebook 的五折 OOF 为
`0.966520498`，单模不如最强 TabM/Lookup，但它与本项目最终预测的 Spearman 只有
约 `0.975`，因此有融合价值。

公开产物使用外层 seed 21，而项目二层 stack 使用 seed 42。不同折的 OOF 不能直接
作为新列送进 seed-42 Logistic meta-model，否则会制造跨层依赖。因此先用固定 rank
blend 验证多样性，再固定公开源码哈希并按项目 seed 42 重训，使新模型能合法进入
global/regime meta-model：

```powershell
python scripts/download_public_spline.py --include-source
python scripts/train_spline_seed42.py
python scripts/build_spline_rank_blend.py `
  --baseline-dir outputs/leaderboard_v5_spline_seed42_stack `
  --weight 0.04 `
  --output-dir outputs/leaderboard_v5_dual_spline_rank_w04
```

### 值得参考：生成器痕迹

来源：

- [模型容量与首位小数讨论](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/734990)
- [生成器特征与消融讨论](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/733541)

本地复算确认 `daily_screen_time_hours` 首位小数 0–9 的标签率约在
`65.1%–73.7%` 之间波动，每个桶有约 5.1–6.8 万行。这不是微小样本噪声。

新特征视图包括：

- 六个时长字段的首位小数、有效小数长度和小数部分；
- `daily - (social + gaming + work)` 残差、绝对残差、占比和越界标记；
- CatBoost 专用的原始数值字符串层级，连续值仍同时保留。

这些特征放在独立 `artifact` / `artifact_cat` 视图中，不改变 V3 原始特征行为。

### 不采用：伪装成神经网络的公开提交融合

[0.97101 来源核验讨论](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735404)
显示，所谓 NN 在最终融合中的权重只有 `1e-6`，实际分数来自两个已有公开提交的
复制。它没有提供可复现的新 OOF 信号。

### 不采用：Reverse Micro-Sorting

[Public LB 探测讨论](https://www.kaggle.com/competitions/playground-series-s6e8/discussion/735339)
通过把预测分成 500 个小桶，再反转桶内 LGBM 顺序，把公开分从 `0.97100` 推到
`0.97101`。作者也明确将其视为 Public LB overfitting。Private 使用不同样本，这种
操作没有可靠迁移依据。

### 只作榜单情报：公开提交 rank blend

[Public 0.97099 Notebook](https://www.kaggle.com/code/daniilkrasnovvv/s6e8-top-1-public-0-97099/notebook)
只是三个公开 submission 的等权 rank average，没有新的训练模型或 OOF。它可以解释
为什么多人迅速接近 `0.971`，但不能替代本项目的同折验证。

## 本地实验结果

### 新一级模型

| 模型 | 特征视图 | OOF ROC-AUC | 说明 |
|---|---|---:|---|
| `cat_artifact_levels` | `artifact_cat` | 0.967538194 | 连续值 + 精确类别层级，最强新单模 |
| `spline_seed42` | 公开样条架构，项目固定五折 | 0.967136226 | RTX 5090 D，约 58.3 分钟 |
| `xgb_generator_artifacts` | `engineered_artifact` | 0.966517615 | 工程特征 + 生成器痕迹 |
| `lgb_generator_artifacts` | `artifact` | 0.966103570 | 生成器痕迹，单模较弱 |

CatBoost 比现有公开库的 `latwide_cat=0.96718` 高约 `0.00036`，比
`digit_cat=0.96666` 高约 `0.00088`。

样条训练严格保持公开 Notebook 的 checkpoint 规则：每个 outer-valid 折同时用于
early stopping 和最终 OOF 预测。这与项目现有一级模型口径一致，但会带来轻微的
checkpoint-selection 乐观偏差；训练指标文件已显式记录
`outer_valid_checkpoint_selection=true`，解读 `1e-5` 量级增益时不能忽略。

### 完整 stack

| 阶段 | V3 86 列 | V5 89 列 | V5 90 列 |
|---|---:|---:|---:|
| Global logistic meta | 0.969708243 | 0.969732476 | 0.969742524 |
| Missingness regime meta | 0.969733184 | 0.969756587 | 0.969762769 |
| Rank + band 最终结果 | 0.969750802 | 0.969774678 | **0.969781998** |

最终相对 V3 的逐折增量为：

```text
fold 0  +0.000020233
fold 1  +0.000019070
fold 2  +0.000017434
fold 3  +0.000030461
fold 4  +0.000032285
```

5/5 折同向，说明它不是由单折偶然提升驱动。新旧 test 预测 Spearman 为
`0.999948`；整体排序变化很小，但方向在 OOF 上稳定。

seed-42 样条进入 90 列后，相对 89 列最终结果再提升 `+0.000007319`。逐折增量为
`+0.000007993 / +0.000011341 / -0.000004763 / -0.000000172 /
+0.000013900`，只有 3/5 折正向，因此它是有效但偏弱的增量，不单独取代 89 列稳健
候选。新列在 global meta 中的系数为 `+0.06394`；90/89 test Spearman 为
`0.99999264`。

### 双 seed 样条固定 rank blend

公开 seed-21 样条仍有独立随机折多样性。在 90 列最终结果上加入 4% 公开样条 rank：

```text
0.969781998 -> 0.969788823，增量 +0.000006826
```

逐折增量为 `+0.000003292 / +0.000013854 / +0.000000527 /
+0.000004262 / +0.000010726`，5/5 正向。3%–5% 均位于稳定平台；4% 比 5% 只高
`0.000000083`，选择 4% 是因为所有折都不回退，而不是继续细搜 OOF。公开样条与
90 列结果的 OOF/Test Spearman 为 `0.974565 / 0.983654`。由于它使用 seed 21，仍只
作固定权重直接融合，不进入 seed-42 二层 LR。

## 提交策略

优先级建议：

1. `outputs/leaderboard_v5_dual_spline_rank_w04/submission.csv`：当前最高 OOF 冲榜提交；
2. `outputs/leaderboard_v4_artifact_stack/submission.csv`：5/5 折稳定提升的同折候选；
3. `outputs/leaderboard_v5_spline_seed42_stack/submission.csv`：严格 seed-42 90 列备选；
4. 继续保留 `outputs/leaderboard_v3_combined/submission.csv` 作为架构回退。

三个新 CSV 都已验证：296,302 行、列顺序 `id,addicted_label`、ID 与官方模板逐行
一致、无 NaN/Inf、预测范围位于 `[0,1]`。

## 复现命令

```powershell
# 训练 3 个 seed-42 生成器痕迹模型
python run.py --config configs/leaderboard_v4_artifacts.yaml train

# 与现有 86 列一起做完整 89 列 stack
python run.py --config configs/leaderboard_v4_artifact_stack.yaml stack-public

# 下载公开样条 OOF/Test 和固定哈希源码
python scripts/download_public_spline.py --include-source

# WSL + CUDA 按项目 seed42 五折重训公开样条架构
python scripts/train_spline_seed42.py

# 将 seed42 样条作为第 90 列运行完整 stack
python run.py --config configs/leaderboard_v5_spline_seed42_stack.yaml stack-public

# 再加入 4% 独立 seed21 样条 rank
python scripts/build_spline_rank_blend.py `
  --baseline-dir outputs/leaderboard_v5_spline_seed42_stack `
  --weight 0.04 `
  --output-dir outputs/leaderboard_v5_dual_spline_rank_w04

# 回归测试
python -m unittest discover -s tests -p "test_*.py" -v
```

## 下一步

样条 seed-42 重训和完整 meta 接入已经完成，可信边际只有 `+0.000007319`。下一步若
继续冲分，应寻找新的低相关函数族或新的公开 OOF，而不是在 3%–5% 样条权重平台上
细搜百万分位。几乎都跑到最大迭代数的 `cat_artifact_levels` 可以延长，但同类
CatBoost 的 stack 边际收益预计小于新函数族。
