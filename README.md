# AirRank：基于评论文本的 Airbnb 房源排序

AirRank 是一个两阶段房源排序项目。第一阶段使用 TextCNN 与 BiLSTM，从 Airbnb 评论文本中预测评论级满意度分数；第二阶段在房源级别聚合文本预测、平均评分、评分稳定性和评论数量，生成最终房源排名。

## 项目特点

- 按 `listing_id` 划分训练集、验证集和测试集，避免同一房源的信息泄漏到不同数据集。
- 使用双向 LSTM 建模评论的上下文语义。
- 使用尺寸为 3、4、5 的多卷积核 TextCNN 提取局部短语特征。
- 使用 SmoothL1Loss 减少少量大误差样本对训练的影响。
- 使用验证集 MAE 和 Early Stopping 选择最佳模型。
- 将评论级预测聚合为房源级排序分数，并与平均评分基线进行比较。

## 项目结构

```text
airrank-listing-ranking/
├─ src/
│  └─ airrank_listing_ranker.py     # 训练、评估与排序主程序
├─ data/
│  └─ README.md                     # 数据格式和放置说明
├─ outputs/
│  ├─ best_cnn_bilstm_listing_ranker.pt
│  ├─ listing_scores.csv
│  ├─ model_topk_listings.csv
│  ├─ baseline_topk_listings.csv
│  ├─ ranking_metrics.csv
│  ├─ summary_metrics.csv
│  ├─ training_configuration.csv
│  ├─ training_history.csv
│  └─ training_curve.png
├─ requirements.txt
├─ .gitignore
└─ README.md
```

训练数据、完整评论文本和评论 ID 不包含在公开仓库中。

## 环境要求

- Python 3.10 或更高版本
- 推荐使用虚拟环境
- 支持 CPU；安装兼容 CUDA 的 PyTorch 后可自动使用 GPU

## 安装方法

在项目根目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 PowerShell 阻止激活脚本，可以只对当前窗口执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 准备数据

默认情况下，程序读取：

```text
data/datasets.zip
```

压缩包内部必须包含：

```text
step1_review_dataset.jsonl
```

JSONL 中每条记录需要包含：

```text
listing_id, review_id, review_text, rating, date
```

详细说明见 [`data/README.md`](data/README.md)。

如果数据位于其他位置，可以通过环境变量指定：

```powershell
$env:AIRRANK_DATASET_ZIP = "D:\data\datasets.zip"
```

## 运行项目

```powershell
python .\src\airrank_listing_ranker.py
```

程序默认把模型、指标和图表保存到 `outputs/`。也可以指定其他输出目录：

```powershell
$env:AIRRANK_OUTPUT_DIR = "D:\airrank-output"
python .\src\airrank_listing_ranker.py
```

## 模型配置

| 参数 | 设置 |
|---|---:|
| 词表上限 | 20,000 |
| 最短词频 | 2 |
| 最大文本长度 | 180 |
| Embedding 维度 | 128 |
| BiLSTM 隐藏维度 | 128 |
| CNN 卷积核尺寸 | 3、4、5 |
| 每种卷积核数量 | 64 |
| Dropout | 0.35 |
| Batch Size | 32 |
| 优化器 | Adam |
| 学习率 | 0.001 |
| 最大 Epoch | 12 |
| Early Stopping Patience | 3 |

## 排序方法

模型首先预测每条评论的满意度分数，然后按房源聚合以下指标：

- 评论数量
- 平均真实评分
- 平均预测评分
- 预测评分标准差
- 稳定性
- 置信度

基础排序分数为：

```text
listing_score = 0.55 × 预测评分 + 0.30 × 真实评分 + 0.15 × 稳定性
```

最终分数再根据评论数量计算的置信度进行调整。

## 已有实验结果

| 指标 | 结果 |
|---|---:|
| Test MAE | 0.0391 |
| Test RMSE | 0.2436 |
| Test R² | 0.0265 |

数据中五星评论占比较高，因此较低的 MAE 并不代表模型对所有评分区间都有同等区分能力。R² 较低也说明模型对少数非五星评论的泛化能力仍有提升空间。

## 输出文件说明

- `best_cnn_bilstm_listing_ranker.pt`：最佳模型权重
- `listing_scores.csv`：全部房源聚合分数
- `model_topk_listings.csv`：模型排序的 Top-K 房源
- `baseline_topk_listings.csv`：平均评分基线的 Top-K 房源
- `ranking_metrics.csv`：模型与基线的排序对比指标
- `summary_metrics.csv`：评论级预测指标
- `training_curve.png`：训练曲线
- `training_configuration.csv`：训练配置

包含完整评论文本和评论 ID 的评论级输出只保存在本地，并通过 `.gitignore` 排除，不在公开仓库中发布。

## 隐私与使用说明

- 数据集和完整评论文本不随项目公开。
- 使用数据前请确认数据来源、平台条款和相应许可。
- 仓库未附带开源许可证，默认不授予商业使用或再分发权利。
- 本项目用于机器学习研究、课程实验和工程展示。

