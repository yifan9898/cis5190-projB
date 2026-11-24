"""
提交用 preprocess.py（自包含，不依赖项目里其他文件）。

prepare_data(csv_path) -> (X, y)
  X: list[str]  截断后的标题词，小写、空格分隔，例如 "trump signs border bill"
  y: list[str]  'FoxNews' / 'NBC'；测试 CSV 里链接没有域名 → ''（后端不用 y）
保证：每一行输出一个元素，顺序不变，绝不删行。
"""
from __future__ import annotations

import csv
import re
from typing import List, Tuple
from urllib.parse import unquote, urlparse


def reduce_url(url: str) -> str:
    """与官方 eval 的 _hide_answers 完全一致（已在 4800 条上逐条验证）。对已截断的链接是幂等的。"""
    parsed = urlparse(str(url).strip())
    segments = [s for s in parsed.path.split("/") if s]
    last = unquote(segments[-1]) if segments else ""
    last = re.sub(r"\.(html?|shtml|php)$", "", last, flags=re.IGNORECASE)
    words = [w for w in re.split(r"[-_]+", last) if w and not re.search(r"\d", w)]
    return "/" + "-".join(words)


def url_to_text(url: str) -> str:
    return reduce_url(url).strip("/").replace("-", " ").lower()


def label_from_url(url: str) -> str:
    host = urlparse(str(url).strip()).netloc.lower()
    if "foxnews" in host:
        return "FoxNews"
    if "nbcnews" in host:
        return "NBC"
    return ""


def prepare_data(csv_path: str) -> Tuple[List[str], List[str]]:
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0].strip().lower() == "url":
        rows = rows[1:]
    X: List[str] = []
    y: List[str] = []
    for row in rows:
        if not row:  # 完全空白的行（csv.reader 给出 []），pandas 也不把它算作一行
            continue
        url = row[0]
        X.append(url_to_text(url))
        y.append(label_from_url(url))
    return X, y
