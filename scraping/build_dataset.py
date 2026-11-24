"""
把 data/raw/fox/*.csv 和 data/raw/nbc/*.csv 清洗合并成训练集，并打印数据报告。

用法：
  python scraping/build_dataset.py                          # 默认 2024-01 ~ 2025-12，按月平衡
  python scraping/build_dataset.py --balance total          # 只做总量平衡，不按月
  python scraping/build_dataset.py --start 2023-01 --end 2026-06

输出：
  data/scraped_clean.csv   爬到的数据（url,label,source,month,section,reduced,n_words）
  data/combined_train.csv  只有 url 一列 = 官方 train_split + 爬取数据，可直接喂给现有 prepare_data / 训练脚本
  data/dataset_report.txt  数据报告（写报告/做图用）
防泄露：与官方 url_train.csv 重复的、以及截断后文本和 val_split 里任何一条相同的，都会被剔除。
"""
from __future__ import annotations

import argparse
import glob
import io
from contextlib import redirect_stdout
from urllib.parse import urlparse, urlunparse

import pandas as pd

from common import ROOT  # noqa: F401  (也会把 src/ 加进 sys.path)
from textnorm import label_from_url, reduce_url, reduced_words, section_of

DATA = ROOT / "data"


def canon(url: str) -> str:
    """统一成 https://www.<host>/<path>，去掉 query/fragment/末尾斜杠，用于去重。"""
    p = urlparse(str(url).strip())
    host = p.netloc.lower()
    if host in ("foxnews.com", "nbcnews.com"):
        host = "www." + host
    return urlunparse(("https", host, p.path.rstrip("/"), "", "", ""))


def load_raw() -> pd.DataFrame:
    frames = []
    fox_files = sorted(glob.glob(str(DATA / "raw" / "fox" / "*.csv")))
    if fox_files:
        f = pd.concat([pd.read_csv(x, dtype=str) for x in fox_files], ignore_index=True)
        f["month"] = f["lastmod"].fillna("").str[:7]
        frames.append(f[["url", "month"]])
    nbc_files = sorted(glob.glob(str(DATA / "raw" / "nbc" / "*.csv")) + glob.glob(str(DATA / "raw" / "nbc_video" / "*.csv")))
    if nbc_files:
        n = pd.concat([pd.read_csv(x, dtype=str) for x in nbc_files], ignore_index=True)
        n["month"] = n["year"] + "-" + n["month"].str.zfill(2)
        frames.append(n[["url", "month"]])
    if not frames:
        raise SystemExit("data/raw/ 下还没有数据，先跑 fox_sitemap.py / nbc_archive.py")
    return pd.concat(frames, ignore_index=True)


def describe(df: pd.DataFrame, name: str) -> None:
    print(f"\n===== {name}: {len(df)} 条 =====")
    print(df["label"].value_counts().to_string())
    print("\n截断后词数（均值 / 中位数）:")
    print(df.groupby("label")["n_words"].agg(["mean", "median"]).round(2).to_string())
    print("\n各来源 Top 12 栏目:")
    for lab, g in df.groupby("label"):
        top = g["section"].value_counts(normalize=True).head(12).round(3)
        print(f"  {lab}: " + ", ".join(f"{k}={v}" for k, v in top.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--balance", choices=["month", "total", "none"], default="month")
    ap.add_argument("--min-words", type=int, default=2, help="截断后少于这么多词的样本丢掉")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-sections", nargs="*", default=["live-news"],
                    help="这些栏目不受时间窗和按月平衡限制，全部保留（稀缺栏目，如 Fox live-news）")
    args = ap.parse_args()

    raw = load_raw()
    n_raw = len(raw)
    raw["url"] = raw["url"].map(canon)
    raw["label"] = raw["url"].map(label_from_url)
    raw = raw[raw["url"].map(lambda u: urlparse(u).netloc in ("www.foxnews.com", "www.nbcnews.com"))]
    raw = raw.drop_duplicates("url")
    n_dedup = len(raw)
    raw["section"] = raw["url"].map(section_of)
    keep = raw["section"].isin(args.keep_sections)
    raw = raw[keep | ((raw["month"] >= args.start) & (raw["month"] <= args.end))]
    n_window = len(raw)

    # 防泄露
    official = pd.read_csv(DATA / "url_train.csv")["url"].map(canon)
    val_text = set(pd.read_csv(DATA / "val_split.csv")["url"].map(reduce_url))
    raw = raw[~raw["url"].isin(set(official))]
    raw["reduced"] = raw["url"].map(reduce_url)
    raw = raw[~raw["reduced"].isin(val_text)]
    n_noleak = len(raw)

    raw["n_words"] = raw["url"].map(lambda u: len(reduced_words(u)))
    raw = raw[raw["n_words"] >= args.min_words]
    # 截断后文本完全相同、标签相同的只留一条；文本相同但标签冲突的全部丢掉（无法学习的噪声）
    conflict = raw.groupby("reduced")["label"].nunique()
    raw = raw[~raw["reduced"].isin(conflict[conflict > 1].index)].drop_duplicates("reduced")
    n_clean = len(raw)

    kept = raw[raw["section"].isin(args.keep_sections)]
    raw = raw[~raw["section"].isin(args.keep_sections)]
    if args.balance == "month":
        parts = []
        for _, g in raw.groupby("month"):
            k = g["label"].value_counts().min() if g["label"].nunique() == 2 else 0
            parts += [x.sample(k, random_state=args.seed) for _, x in g.groupby("label")] if k else []
        bal = pd.concat(parts) if parts else raw.iloc[:0]
    elif args.balance == "total":
        k = raw["label"].value_counts().min()
        bal = pd.concat([x.sample(k, random_state=args.seed) for _, x in raw.groupby("label")])
    else:
        bal = raw
    bal = pd.concat([bal, kept]).assign(source="scraped").sort_values(["month", "label"])

    cols = ["url", "label", "source", "month", "section", "reduced", "n_words"]
    bal[cols].to_csv(DATA / "scraped_clean.csv", index=False)
    train_split = pd.read_csv(DATA / "train_split.csv")["url"]
    pd.DataFrame({"url": pd.concat([train_split, bal["url"]], ignore_index=True)}).to_csv(
        DATA / "combined_train.csv", index=False
    )

    off = pd.read_csv(DATA / "url_train.csv")
    off["label"] = off["url"].map(label_from_url)
    off["n_words"] = off["url"].map(lambda u: len(reduced_words(u)))
    off["section"] = off["url"].map(section_of)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("清洗漏斗:")
        print(f"  原始抓取 {n_raw} → 去重 {n_dedup} → 时间窗 {args.start}~{args.end}: {n_window}"
              f" → 去除与官方/验证集重叠 {n_noleak} → 词数≥{args.min_words} 且去冲突 {n_clean}"
              f" → 平衡({args.balance}) {len(bal) - len(kept)} + 稀缺栏目全保留 {len(kept)} ({', '.join(args.keep_sections)}) = {len(bal)}")
        print(f"combined_train.csv = train_split {len(train_split)} + scraped {len(bal)} = {len(train_split) + len(bal)}")
        describe(off, "官方 url_train.csv（参照分布）")
        describe(bal, "爬取数据（平衡后）")
        print("\n每月样本数（平衡后）:")
        print(bal.pivot_table(index="month", columns="label", values="url", aggfunc="count").fillna(0).astype(int).to_string())
    report = buf.getvalue()
    print(report)
    (DATA / "dataset_report.txt").write_text(report)


if __name__ == "__main__":
    main()
