"""
NBC News 链接收集：遍历按月归档页 https://www.nbcnews.com/archive/{articles|videos}/<year>/<month>[/<page>]

用法（项目根目录下，已激活 .venv）：
  python scraping/nbc_archive.py --probe                       # 试抓 2024 年 1 月第 1 页，看解析对不对
  python scraping/nbc_archive.py --start 2024-01 --end 2025-12 # 全量（文章）
  python scraping/nbc_archive.py --kind videos --probe          # 视频归档
  python scraping/nbc_archive.py --kind videos --start 2024-01 --end 2025-12

每个 (年, 月, 页) 存成 data/raw/nbc/<year>_<mm>_p<page>.csv；已存在的跳过 → 可中断重跑。
某月翻页直到：404 / 页面没有新链接 / 达到 --max-pages-per-month。
"""
from __future__ import annotations

import argparse
from datetime import date
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from common import RAW_DIR, polite_get, write_rows

BASE_TMPL = "https://www.nbcnews.com/archive/{kind}"
FIELDS = ["url", "year", "month", "page"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
# 这些一级路径是导航/归档/作者页，不是文章
_SKIP_PREFIX = ("archive", "author", "authors", "information", "newsletters", "search", "app", "pages", "datagraphics")


def is_article_link(href: str) -> bool:
    p = urlparse(href)
    if p.netloc not in ("www.nbcnews.com", "nbcnews.com"):
        return False
    segs = [s for s in p.path.split("/") if s]
    if len(segs) < 2 or segs[0].lower() in _SKIP_PREFIX:
        return False
    last = segs[-1]
    # 文章 slug 至少含一个连字符（排除 /politics/congress 这种栏目页）
    return "-" in last and len(last) > 8


def extract_links(page_html: str, page_url: str) -> list[str]:
    soup = BeautifulSoup(page_html, "html.parser")
    main = soup.find("main") or soup  # 优先只看正文区，避免抓到页头页脚导航
    seen, out = set(), []
    for a in main.find_all("a", href=True):
        href = urljoin(page_url, a["href"]).split("#")[0].split("?")[0]
        if is_article_link(href) and href not in seen:
            seen.add(href)
            out.append(href)
    return out


def month_iter(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def page_url(y: int, m: int, page: int, kind: str = "articles") -> str:
    u = f"{BASE_TMPL.format(kind=kind)}/{y}/{MONTHS[m - 1]}"
    return u if page == 1 else f"{u}/{page}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--max-pages-per-month", type=int, default=40)
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--kind", choices=["articles", "videos"], default="articles",
                    help="articles → data/raw/nbc/；videos → data/raw/nbc_video/（now / nightly-news / meet-the-press 等视频页）")
    args = ap.parse_args()
    OUT_DIR = RAW_DIR / ("nbc" if args.kind == "articles" else "nbc_video")

    if args.probe:
        y, m = map(int, args.start.split("-"))
        for pg in (1, 2):
            u = page_url(y, m, pg, args.kind)
            r = polite_get(u, delay=0 if pg == 1 else args.delay)
            if r is None:
                print(f"[probe] {u} → 没拿到（404 或被拒）")
                continue
            links = extract_links(r.text, u)
            print(f"[probe] {u} → HTML {len(r.text)} 字节, 解析出 {len(links)} 条文章链接")
            for l in links[:8]:
                print("   ", l)
        return

    total = 0
    today = date.today()
    for y, m in month_iter(args.start, args.end):
        if (y, m) > (today.year, today.month):
            break
        prev: set[str] = set()
        for pg in range(1, args.max_pages_per_month + 1):
            out = OUT_DIR / f"{y}_{m:02d}_p{pg:02d}.csv"
            if out.exists():
                continue
            u = page_url(y, m, pg, args.kind)
            r = polite_get(u, delay=args.delay)
            if r is None:
                break
            links = extract_links(r.text, u)
            new = [l for l in links if l not in prev]
            if not new:
                break
            prev.update(new)
            write_rows(out, [{"url": l, "year": y, "month": m, "page": pg} for l in new], FIELDS)
            total += len(new)
            print(f"{y}-{m:02d} p{pg}: {len(new)} 条  累计 {total}", flush=True)
    print(f"完成。文件在 {OUT_DIR}")


if __name__ == "__main__":
    main()
