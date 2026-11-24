"""
Fox live-news（直播博客）URL 收集：Fox 官网 sitemap 不收录、也没有列表页，
所以用 Internet Archive 的公开 CDX 接口查询"历史上被存档过的 foxnews.com/live-news/* 链接"。
只取 URL 字符串和首次存档时间（近似发布时间），不抓任何页面内容。

用法：
  python scraping/fox_livenews_cdx.py --probe     # 只查 20 条看看格式
  python scraping/fox_livenews_cdx.py             # 全量，结果存 data/raw/fox/livenews_cdx.csv

输出列与 fox_sitemap.py 一致（url, lastmod, sitemap），build_dataset.py 会自动读入。
"""
from __future__ import annotations

import argparse
import json
import re
from urllib.parse import urlencode, urlparse

from common import RAW_DIR, polite_get, write_rows

CDX = "https://web.archive.org/cdx/search/cdx"
OUT = RAW_DIR / "fox" / "livenews_cdx.csv"
FIELDS = ["url", "lastmod", "sitemap"]
_SLUG_OK = re.compile(r"^[a-z0-9][a-z0-9\-]*$")


def query(limit: int, resume_key: str | None):
    params = {
        "url": "foxnews.com/live-news/",
        "matchType": "prefix",
        "fl": "original,timestamp",
        "collapse": "urlkey",          # 同一个 URL 只返回一次（最早的存档）
        "filter": "statuscode:200",
        "output": "json",
        "limit": str(limit),
        "showResumeKey": "true",
    }
    if resume_key:
        params["resumeKey"] = resume_key
    r = polite_get(f"{CDX}?{urlencode(params)}", delay=2.0, timeout=120)
    if r is None or not r.text.strip():
        return [], None
    data = json.loads(r.text)
    rows, next_key = [], None
    # 格式：[header, row, row, ..., [], [resumeKey]]
    for item in data[1:]:
        if not item:
            continue
        if len(item) == 1:
            next_key = item[0]
            continue
        rows.append(item)
    return rows, next_key


def clean(original: str) -> str | None:
    p = urlparse(original if "://" in original else "https://" + original)
    segs = [s for s in p.path.split("/") if s]
    if len(segs) != 2 or segs[0] != "live-news":
        return None
    slug = segs[1].lower()
    if not _SLUG_OK.match(slug) or "-" not in slug:
        return None
    return f"https://www.foxnews.com/live-news/{slug}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--page-size", type=int, default=5000)
    args = ap.parse_args()

    if args.probe:
        rows, key = query(20, None)
        print(f"[probe] 返回 {len(rows)} 条，有下一页: {bool(key)}")
        for orig, ts in rows:
            print("   ", ts, orig, "→", clean(orig))
        return

    seen: dict[str, str] = {}
    key = None
    while True:
        rows, key = query(args.page_size, key)
        for orig, ts in rows:
            u = clean(orig)
            if u and (u not in seen or ts < seen[u]):
                seen[u] = ts
        print(f"累计 {len(seen)} 条唯一 live-news URL", flush=True)
        if not key or not rows:
            break
    out = [{"url": u, "lastmod": f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}", "sitemap": "cdx"} for u, ts in sorted(seen.items())]
    write_rows(OUT, out, FIELDS)
    years: dict[str, int] = {}
    for r in out:
        years[r["lastmod"][:4]] = years.get(r["lastmod"][:4], 0) + 1
    print(f"完成：{len(out)} 条 → {OUT}\n按首次存档年份: {dict(sorted(years.items()))}")


if __name__ == "__main__":
    main()
