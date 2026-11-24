"""
全项目唯一的 URL 截断实现。逻辑与官方 eval_project_b.py 的 _hide_answers 完全一致：
只保留路径最后一段 → 去 .html 等后缀 → 按 - / _ 切词 → 丢掉含数字的词。
训练、爬虫清洗、提交的 preprocess.py 都必须用这一份（提交时把函数复制进去）。
"""
from __future__ import annotations

import re
from urllib.parse import unquote, urlparse


def reduce_url(url: str) -> str:
    """完整链接或已截断链接 → '/word-word-word'（与排行榜测试输入同格式）"""
    parsed = urlparse(str(url).strip())
    segments = [s for s in parsed.path.split("/") if s]
    last = unquote(segments[-1]) if segments else ""
    last = re.sub(r"\.(html?|shtml|php)$", "", last, flags=re.IGNORECASE)
    words = [w for w in re.split(r"[-_]+", last) if w and not re.search(r"\d", w)]
    return "/" + "-".join(words)


def reduced_words(url: str) -> list[str]:
    return [w for w in reduce_url(url).strip("/").split("-") if w]


def label_from_url(url: str) -> str:
    host = urlparse(str(url).strip()).netloc.lower()
    if "foxnews" in host:
        return "FoxNews"
    if "nbcnews" in host:
        return "NBC"
    return ""


def section_of(url: str) -> str:
    segs = [s for s in urlparse(str(url).strip()).path.split("/") if s]
    return segs[0] if len(segs) > 1 else ""
