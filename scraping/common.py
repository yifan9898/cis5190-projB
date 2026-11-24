"""爬虫公共工具：带限速、重试、退避的 HTTP 请求，以及断点续爬用的 checkpoint 读写。"""
from __future__ import annotations

import csv
import os
import random
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT / "src"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_session = requests.Session()
_session.headers.update(HEADERS)


def polite_get(url: str, *, delay: float = 1.5, retries: int = 4, timeout: int = 30) -> requests.Response | None:
    """GET 一个页面。每次请求前 sleep(delay±30%)；429/5xx/网络错误指数退避重试。
    404 返回 None（用来判断分页结束），其他失败在重试耗尽后返回 None 并打印原因。"""
    for attempt in range(retries):
        time.sleep(delay * random.uniform(0.7, 1.3))
        try:
            r = _session.get(url, timeout=timeout)
        except requests.RequestException as e:
            wait = 5 * 2**attempt
            print(f"  [net error] {e!r} → 等 {wait}s 重试", flush=True)
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        if r.status_code in (403, 429) or r.status_code >= 500:
            wait = 15 * 2**attempt
            print(f"  [HTTP {r.status_code}] {url} → 等 {wait}s 重试", flush=True)
            time.sleep(wait)
            continue
        print(f"  [HTTP {r.status_code}] {url} → 跳过", flush=True)
        return None
    print(f"  [give up] {url}", flush=True)
    return None


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    """原子写一个 checkpoint 文件（先写 .tmp 再改名），中途被打断也不会留下半个文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
