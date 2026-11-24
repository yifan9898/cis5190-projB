# data/

数据文件不进 git，按下面步骤重新生成（在项目根目录、激活 `.venv` 后执行）。

| 文件 | 来源 |
|---|---|
| `url_train.csv` | 课程提供（Drive 上的 Proj B Template），4800 条，单列 `url` |
| `train_split.csv` / `val_split.csv` | 从 `url_train.csv` 按标签分层 80/20 切分，`random_state=42` |
| `raw/fox/articles_*.csv` | `python scraping/fox_sitemap.py --start 2024-01` |
| `raw/fox/livenews_cdx.csv` | `python scraping/fox_livenews_cdx.py`（Internet Archive CDX，仅 URL） |
| `raw/nbc/*.csv` | `python scraping/nbc_archive.py --start 2024-01 --end 2025-12` |
| `raw/nbc_video/*.csv` | `python scraping/nbc_archive.py --kind videos --start 2024-01 --end 2025-12` |
| `scraped_clean.csv` / `combined_train.csv` / `dataset_report.txt` | `python scraping/build_dataset.py` |

切分代码：

```python
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("data/url_train.csv")
df["label"] = df.url.str.contains("foxnews").map({True: "FoxNews", False: "NBC"})
tr, va = train_test_split(df, test_size=0.2, random_state=42, stratify=df.label)
tr[["url"]].to_csv("data/train_split.csv", index=False)
va[["url"]].to_csv("data/val_split.csv", index=False)
```
