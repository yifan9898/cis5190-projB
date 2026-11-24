# CIS 5190 Project B — News Source Classification

根据新闻链接中截断后的标题词（无域名、栏目、数字），判断文章来自 **Fox News** 还是 **NBC News**。

## 当前结果
| 版本 | val（960 条） | 5 折 CV |
|---|---|---|
| 官方 baseline（哈希 EmbeddingBag + MLP） | 73.75% | — |
| TF-IDF（词 1–2gram + 字符 2–5gram）+ LR，仅官方数据 | 82.50% | 80.8% |
| + 爬取数据 + 按栏目重加权 | 86.04% | — |
| + NBC 视频 | 88.12% | 86.3% |
| + slug 格式元特征 | 88.54% | 87.1% |
| + Fox live-news（当前提交版） | 88.75% | 87.4% |

## 目录结构
```
src/textnorm.py            URL 截断（与官方 eval 的 _hide_answers 逐条一致），全项目唯一实现
scraping/                  数据采集与清洗
  common.py                限速、重试、断点续爬
  fox_sitemap.py           Fox：sitemap(articles) 分页
  fox_livenews_cdx.py      Fox live-news：Internet Archive CDX（仅 URL）
  nbc_archive.py           NBC：/archive/{articles,videos}/年/月
  build_dataset.py         去重、时间窗、防泄露、按月平衡 → scraped_clean.csv
experiments/
  train_linear.py          训练 TF-IDF 线性模型，导出 torch state_dict，记录到 results.csv
  results.csv              所有实验记录
submission/                排行榜提交文件（model.py, preprocess.py；model.pt 需本地训练生成）
template/                  课程提供的模板、官方 baseline 和本地评测脚本
scripts/                   一键运行脚本
data/                      不进 git，见 data/README.md
```

## 复现
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch scikit-learn pandas requests beautifulsoup4

# 1. 数据（见 data/README.md）
python scraping/fox_sitemap.py --start 2024-01
python scraping/fox_livenews_cdx.py
python scraping/nbc_archive.py --start 2024-01 --end 2025-12
python scraping/nbc_archive.py --kind videos --start 2024-01 --end 2025-12
python scraping/build_dataset.py

# 2. 训练 + 导出 submission/model.pt + 官方 eval 验证
bash scripts/run_tfidf.sh --extra data/scraped_clean.csv --section-reweight --name tfidf_meta_lr_livenews
```

## 模型
- 特征：词 1–2gram TF-IDF、字符 2–5gram TF-IDF、slug 格式元特征（长度、末词长度、首/末词、孤立的 `s`），各组 L2 归一化后哈希到 2^19 桶
- 分类器：逻辑回归（C=3）；官方样本权重 ×5，爬取样本按栏目重加权到官方分布
- 提交形式：`EmbeddingBag(2^19, 2, sum)` + bias + idf buffer，单个 `nn.Module`，导出时自检与 sklearn 预测一致
