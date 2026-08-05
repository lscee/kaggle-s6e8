# Playground S6E8 冲榜方案 V3

更新日期：2026-08-05

## 结论

当前本地最强、且通过同折 OOF 检查的候选为：

- 主候选：`outputs/leaderboard_v3_combined/submission.csv`
- 稳健候选：`outputs/leaderboard_v3_fm/submission.csv`
- 公开基准后处理候选：`outputs/leaderboard_v3_public/submission_postprocessed.csv`
- 已知公开基准复现：`outputs/leaderboard_v3_public/submission.csv`

主候选的 OOF ROC-AUC 为 **0.969750802**。79 模型稳健候选为
**0.969747202**，只低 0.000003600。两者测试集 Spearman 相关系数为
**0.999989**。

截至 2026-08-05，官方 public leaderboard 第一名约为 **0.97115**。公开的
74 模型方案报告 public LB 约 **0.97084**；若本地 OOF 增量近似转移，V3 的合理预期
约在 **0.9709** 附近，而不是可以保证达到 0.97115。public 榜只覆盖约 20% 测试集，
最后排名由约 80% private 榜决定，因此不应为了几百万分位的 public 提升破坏 CV 稳定性。

## 数据与许可

### 比赛数据

- `data/raw/train.csv`：691,369 行，目标列 `addicted_label`。
- `data/raw/test.csv`：296,302 行。
- `data/raw/sample_submission.csv`：官方提交模板。

### 公开 OOF 库

来源：[S6E8 full OOF library](https://www.kaggle.com/datasets/szymonkapiski/s6e8-oof-library-47-models)

- 版本：v6。
- 许可：CC0 Public Domain。
- 74 组 OOF/test 预测，全部为 float64。
- 固定外层验证：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`。
- `train_keys.parquet`、`test_keys.parquet` 与本地 CSV 的 id 已逐行完全比对。
- `train_keys.parquet` 中的标签与本地训练标签完全一致。
- 74/74 OOF 与 test 数组一一配对，长度正确，无 NaN/Inf。
- 本地复算的单模 AUC 与 manifest 一致；最佳单模为 `naji05 = 0.968814589`。

主要模型族包括 XGBoost、LightGBM、CatBoost、TabM、RealMLP、ResNet、TabNet、
Lookup Transformer、MLP、随机森林，以及多种 lattice target encoding 特征视图。

### Factorization Machine lattice 库

来源：[S6E8 factorization-machine lattice members](https://www.kaggle.com/datasets/raykkretzschmar/s6e8-fm-lattice-blend-members)

- 版本：v5。
- 许可：CC0 Public Domain。
- 5 个全长 FM 成员使用原始 logit 存储；接入时没有错误地当作概率 clip。
- 2 个 band-local 成员只用于区间内重排，不作为全长一级模型。
- parquet id 与本地 train/test 逐行比对。
- 固定外层验证同样为 5-fold、seed 42。

FM 是本轮真正有效的新函数族。它给每个 `(字段, 精确值)` 一个低维向量，用向量内积
表示二阶交互，能够让稀疏 lattice 单元共享统计强度；这与树模型、Transformer 的误差
形态不同。

### 其他来源

- 原始 7,500 行 parent dataset 已测试，但邻居特征降低 OOF，因此未进入最终候选。
- CC0 的 4 模型 Golem OOF 库已审计；其 `74+FM` 增量只有 +0.00000394，未进入主版本。
- 不使用私有数据、标签泄漏、伪标签 OOF 或不同 fold 的 OOF 混合。

## 最终架构

```text
比赛原始特征
  ├─ 74 个同折公开一级模型
  │    ├─ GBDT：XGB / LGBM / CatBoost
  │    ├─ 深度表格：TabM / RealMLP / ResNet / TabNet
  │    ├─ Lookup Transformer
  │    └─ lattice TE、缺失值、残差、rank 等特征视图
  ├─ 5 个 FM lattice logits
  └─ 7 个本地 seed=42 GPU GBDT（仅 86 模型候选）
          ↓
  float64 logit matrix
          ↓
  5-fold LogisticRegression global meta-model
          +
  missingness regime meta-model
    - 原始模型 logits
    - complete-row interactions
    - 4+ missing-row interactions
    - 模型 disagreement interactions
    - mean / std / range 聚合
          ↓
  1/3 global rank + 2/3 regime rank
          ↓
  两个 daily-screen-time band 内部的小权重 FM 重排
          ↓
  submission.csv
```

所有元模型输入保持 float64。公开库的实验显示，在高度相关的 stacking 成员上降为
float32 会改变大量测试行的细微排序，并可能损失约 0.00001 public LB。

## OOF 结果

所有 V3 数字均使用相同的固定 5-fold、seed 42，可直接横向比较。

| 方案 | 模型数 | OOF AUC | 相对 74-global |
|---|---:|---:|---:|
| 74 public，global LR | 74 | 0.969660469 | 基准 |
| 74 public，missingness regime | 74 | 0.969687362 | +0.000026893 |
| 74 public，meta mix + band | 74 + band | 0.969711584 | +0.000051115 |
| 74 + FM，global LR | 79 | 0.969704165 | +0.000043696 |
| 74 + FM，regime | 79 | 0.969729395 | +0.000068926 |
| 74 + FM，meta mix + band | 79 + band | **0.969747202** | **+0.000086733** |
| 74 + FM + 本地 GBDT，global LR | 86 | 0.969708243 | +0.000047774 |
| 74 + FM + 本地 GBDT，regime | 86 | 0.969733184 | +0.000072715 |
| 74 + FM + 本地 GBDT，meta mix + band | 86 + band | **0.969750802** | **+0.000090333** |

### 家族消融

| 新增家族 | global OOF 增量 |
|---|---:|
| 74 + 5 FM | +0.000043696 |
| 74 + 7 本地 GBDT | +0.000002751 |
| 74 + 4 Golem | +0.000005468 |
| 74 + FM 后再加 Golem | +0.000003939 |

FM 贡献了几乎全部新增信号。本地 GBDT 虽然都由 RTX 5090 D 在 seed=42 折上重训，
但其函数族与公共 GBDT 高度重复，增量很小。

### 逐折稳定性

FM 后处理方案相对 74 后处理方案在 5/5 折都提升：

| fold | 74 + band | 74 + FM + band | 增量 |
|---:|---:|---:|---:|
| 0 | 0.969128034 | 0.969174750 | +0.000046716 |
| 1 | 0.969812889 | 0.969851891 | +0.000039002 |
| 2 | 0.969859569 | 0.969907542 | +0.000047972 |
| 3 | 0.970303099 | 0.970343314 | +0.000040216 |
| 4 | 0.969473398 | 0.969484243 | +0.000010844 |

86 模型相对 79 模型也在 5/5 折同向，但每折仅提升约
0.0000015–0.0000052。因此它可以作为主候选，但 79 模型应同时保留为更稳健的 private
候选。

## Missingness 分桶

86 模型的 global → regime 改善在三个桶内均为正：

| 缺失数量 | 行数 | global AUC | regime AUC |
|---|---:|---:|---:|
| 0 | 269,185 | 0.977208740 | 0.977252066 |
| 1–3 | 368,484 | 0.968191005 | 0.968197997 |
| 4+ | 53,700 | 0.930381487 | 0.930444303 |

## 提交文件与校验

| 顺序 | 文件 | 本地 OOF | SHA-256 |
|---:|---|---:|---|
| 1 | `outputs/leaderboard_v3_combined/submission.csv` | 0.969750802 | `C64C430585FABD900FA417FB414855B93F8E70C95A9A7805BA5B06D25DDBA89F` |
| 2 | `outputs/leaderboard_v3_fm/submission.csv` | 0.969747202 | `59AAE65D5BEA4020BE4A1D25174A262F58F7BBD80C3A84C7D9D29DC0748CD336` |
| 3 | `outputs/leaderboard_v3_public/submission_postprocessed.csv` | 0.969711584 | `E5EC688EE49A3B41563E061EA610656EC558252D4B4B5F43DEBB377F61BED9F3` |
| 4 | `outputs/leaderboard_v3_public/submission.csv` | 0.969687362 | `61CDB93B598BB12A812D3EC8DC6724972E98FAD6DA928769685C30D03D464ABC` |

所有提交文件均满足：

- 296,302 行；
- 列顺序严格为 `id,addicted_label`；
- id 与 test.csv 逐行一致；
- 预测全部有限且位于 `[0, 1]`；
- 使用连续排序分数，符合 ROC-AUC 提交要求。

## 复现命令

先在 WSL/GPU 上生成同折本地一级模型：

```powershell
.\run-wsl.ps1 --config /mnt/c/Users/mie/Documents/dice/kaggle-s6e8/configs/leaderboard_v3_seed42.yaml train
```

生成 79 模型稳健候选：

```powershell
.\.venv\Scripts\python.exe run.py --config configs\leaderboard_v3_fm.yaml stack-public
```

生成 86 模型主候选：

```powershell
.\.venv\Scripts\python.exe run.py --config configs\leaderboard_v3_combined.yaml stack-public
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

当前完整测试结果：7/7 通过。

## 实际提交策略

1. 先提交 86 模型主候选。
2. 再提交 79 模型稳健候选。
3. 用公开基准或 74+band 候选校准本账号的 public LB 偏移。
4. 最终 private 选择优先保留 86 与 79，而不是只选 public 分数最高的两个近重复变体。
5. 只有新的模型家族在固定 OOF 上产生至少约 0.00002、且多数折同向时，才替换主候选。

当前机器没有 Kaggle API 凭据，浏览器也没有登录态，因此本项目只生成并校验候选，尚未
替用户执行真实提交。

## 还可能继续提升的方向

剩余空间不在常规 GBDT 调参。已经实测：新增相似 GBDT 只有约 0.000003 增量。更值得投入：

1. 在完全相同的 seed=42 folds 上训练新的低相关架构，例如不同归纳偏置的 FM/DCNv2、
   SAINT 或新的 lookup-attention 变体；先看与 `lookup`、FM 的相关性，再看单模 AUC。
2. 对 4+ missing rows 训练真正的 fold-safe specialist，但必须在完整 pooled OOF 上验证，
   不能只看该桶自身 AUC。
3. 使用 nested stacking 或 repeated outer folds 估计元模型权重方差；不再扩展 C 网格。
4. 获取真实 public LB 反馈后，只进行少量、预先定义的提交；避免对约 20% public 子集过拟合。
