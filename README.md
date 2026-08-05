# Kaggle Playground S6E8 冲榜工程

最新可复现冲榜架构、正式五折结果和最终权重见 [LEADERBOARD_V2.md](LEADERBOARD_V2.md)。当前最终 OOF ROC-AUC 为 **0.965110**，提交文件位于 `outputs/leaderboard_final/submission.csv`。

面向 [Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8) 的可复现实验框架。核心目标不是只训练一个模型，而是建立可靠的泄漏审计、固定 OOF、强表格模型和基于排名的融合流程。

## 当前能力

- 校验 train/test/sample submission 字段和 ID 顺序；
- 固定五折分层交叉验证；
- 单变量及缺失指示 OOF AUC 扫描，识别代理泄漏和异常信号；
- 检查重复样本、冲突标签及 train/test 精确匹配；
- 可选 train/test 对抗验证；
- 手机使用时长、比例、周末差异、睡眠、交互强度和行级缺失统计；
- CatBoost、LightGBM、ExtraTrees、Logistic OOF 训练；
- 每折指标、模型汇总、特征重要性和预测文件自动落盘；
- 基于 OOF AUC 的 greedy rank ensemble；
- 自动输出模型 OOF 排名相关矩阵，判断融合多样性；
- 自动生成 `submission.csv`。

## 目录

```text
kaggle-s6e8/
├─ configs/base.yaml
├─ configs/gpu_wsl.yaml
├─ configs/gpu_smoke.yaml
├─ data/raw/                 # 放 Kaggle CSV，不提交到 Git
├─ outputs/                  # 每个实验的 OOF、指标和 submission
├─ src/s6e8/
├─ tests/test_smoke.py
├─ requirements.txt
├─ run-wsl.ps1
└─ run.py
```

## 1. 准备环境

建议使用 Python 3.10–3.13：

```powershell
cd C:\Users\mie\Documents\dice\kaggle-s6e8
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果暂时不安装 CatBoost 和 LightGBM，审计、ExtraTrees、Logistic 与融合仍然可以运行；配置中的 `skip_missing_models: true` 会跳过缺失模型。

### 当前设备的 WSL2 GPU 环境

当前机器已完成并验证以下环境：

- GPU：NVIDIA GeForce RTX 5090 D，约 32 GB 显存；
- WSL：Ubuntu 22.04；
- Python：`/home/lsc/.venvs/kaggle-s6e8/bin/python`；
- CatBoost 1.2.10：`task_type=GPU`，设备 `0`；
- LightGBM 4.7.0：使用 CUDA 12.9、SM 120 和兼容 Ubuntu 22.04 的 NCCL 编译，`device_type=cuda`；
- `configs/base.yaml` 继续保留 CPU 配置，`configs/gpu_wsl.yaml` 是正式 GPU 配置。

先验证两个模型确实能使用 GPU：

```powershell
cd C:\Users\mie\Documents\dice\kaggle-s6e8
powershell -ExecutionPolicy Bypass -File .\run-wsl.ps1 gpu-check
```

返回 `"status": "ok"`，且同时显示 CatBoost 的 `GPU` 和 LightGBM 的 `cuda` 才算通过。正式五折训练：

```powershell
powershell -ExecutionPolicy Bypass -File .\run-wsl.ps1 `
  --output-dir outputs/gpu_baseline_v1 `
  all --models catboost lightgbm --adversarial
```

快速复验完整真实数据链路可使用两折短配置：

```powershell
powershell -ExecutionPolicy Bypass -File .\run-wsl.ps1 `
  --config configs/gpu_smoke.yaml train --models catboost lightgbm
powershell -ExecutionPolicy Bypass -File .\run-wsl.ps1 `
  --config configs/gpu_smoke.yaml blend
```

`gpu_smoke` 只用于检查训练链路和速度，不应用它代替固定五折冲榜结果。LightGBM CUDA 只支持 Linux 构建，编译方式参考其[官方安装指南](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html)。

## 2. 放入比赛数据

将文件放到：

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

不要改 CSV 行顺序。程序会验证 sample submission 的 ID 是否与 test 一致。

## 3. 第一轮：数据审计

```powershell
python run.py --output-dir outputs/audit_v1 audit --adversarial
```

重点查看：

```text
outputs/audit_v1/audit_summary.json
outputs/audit_v1/audit_univariate_auc.csv
outputs/audit_v1/folds.csv
```

优先回答：

1. 屏幕、社交和周末使用时间的单变量 AUC 是否与完整模型一致；
2. 是否有重复行和标签冲突；
3. 测试集是否有行能在训练集中精确找到；
4. adversarial validation AUC 是否明显高于 0.5；
5. 各字段缺失率的 train/test 差异是否影响验证；
6. `id` 是否值得保留。

## 4. 第一轮模型

先跑强模型和一个线性多样性模型：

```powershell
python run.py --output-dir outputs/baseline_v1 train --models catboost lightgbm logistic
python run.py --output-dir outputs/baseline_v1 blend
```

一条命令跑完整流程：

```powershell
python run.py --output-dir outputs/baseline_v1 all --models catboost lightgbm logistic --adversarial
```

最终提交文件：

```text
outputs/baseline_v1/submission.csv
```

融合诊断文件：

```text
outputs/baseline_v1/ensemble.json
outputs/baseline_v1/oof_rank_correlation.csv
```

## 5. 必做消融实验

### 实验 A：保留全部字段

```powershell
python run.py --output-dir outputs/all_features all --models catboost lightgbm logistic
```

### 实验 B：删除弱类别字段

```powershell
python run.py --output-dir outputs/no_weak_categories --drop-columns gender,stress_level,academic_work_impact all --models catboost lightgbm logistic
```

### 实验 C：加入 ID

```powershell
python run.py --output-dir outputs/with_id --use-id all --models catboost lightgbm logistic
```

### 实验 D：关闭人工特征

```powershell
python run.py --output-dir outputs/raw_features --no-engineered all --models catboost lightgbm logistic
```

只比较统一折号下的 OOF AUC，同时观察各模型预测相关性和融合收益。不要依据一次公开榜结果修改验证结论。

## 6. 快速自检

生成模拟比赛数据：

```powershell
python run.py demo-data --rows 1400 --test-rows 500
python run.py --output-dir outputs/demo all --models extra_trees logistic
```

运行单元测试：

```powershell
python -m unittest discover -s tests -v
```

模拟数据只用于验证代码，不可用于比赛提交。使用真实数据前应删除或覆盖 `data/raw` 中的模拟 CSV。

## 7. 冲榜迭代顺序

1. 锁定固定 OOF，完成泄漏、重复行和分布审计；
2. 建立 CatBoost、LightGBM、Logistic 三个基线；
3. 做弱类别字段、`id`、缺失特征和人工特征消融；
4. 获取并核对疑似 7,500 行公开母数据；
5. 增加母数据最近邻、精确匹配和规则模型；
6. 扩展多 seed、模型参数和低相关模型；
7. 用 OOF greedy rank blend 选择组合；
8. 最后才参考公开榜判断本地验证与测试集是否一致。

详细的问题分析见上级目录的 `Kaggle_S6E8_项目难点与核心.md`，实验台账见 `EXPERIMENTS.md`。
