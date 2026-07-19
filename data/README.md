# 数据集放置说明

训练数据不包含在公开仓库中。

请将 `datasets.zip` 放在本目录下：

```text
data/datasets.zip
```

压缩包内部必须包含：

```text
step1_review_dataset.jsonl
```

每条 JSONL 记录需要包含以下字段：

- `listing_id`：房源 ID
- `review_id`：评论 ID
- `review_text`：评论文本
- `rating`：评分
- `date`：评论日期

也可以不复制数据文件，通过环境变量指定其位置：

```powershell
$env:AIRRANK_DATASET_ZIP = "D:\data\datasets.zip"
python .\src\airrank_listing_ranker.py
```

请确保你有权使用相应数据。不要将包含完整评论文本或个人信息的数据上传到公开仓库。

