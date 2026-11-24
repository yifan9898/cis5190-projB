"""
探测 Fox live-news（直播博客）在 Fox 官网上有没有可遍历的列表入口。
只请求几页，打印每页能解析出多少 /live-news/ 链接、有没有翻页参数。

用法：python scraping/fox_livenews_probe.py
"""
from __future__ import annotations

import re
from urllib.parse import urljoin

from common import polite_get

CANDIDATES = [
    "https://www.foxnews.com/live-news",
    "https://www.foxnews.com/category/live-news",
    "https://www.foxnews.com/category/live-news?page=2",
    "https://www.foxnews.com/category/live-news?page=50",
]


def main() -> None:
    for u in CANDIDATES:
        r = polite_get(u, delay=1.5, retries=2)
        if r is None:
            print(f"[x] {u} → 没拿到（404/被拒）")
            continue
        links = sorted({urljoin(u, h).split("?")[0] for h in re.findall(r'href="([^"]*/live-news/[^"#]+)"', r.text)})
        pages = sorted(set(re.findall(r'[?&]page=(\d+)', r.text)), key=int)[:10]
        print(f"[ok] {u} → HTML {len(r.text)} 字节，/live-news/ 链接 {len(links)} 条，页面里出现的 page= 参数: {pages}")
        for l in links[:6]:
            print("     ", l)


if __name__ == "__main__":
    main()
