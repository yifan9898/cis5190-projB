"""
Fox News 链接收集：遍历 https://www.foxnews.com/sitemap.xml 里的 articles 分页 sitemap。

用法（项目根目录下，已激活 .venv）：
  python scraping/fox_sitemap.py --probe            # 先试抓索引 + 1 页，看结构对不对
  python scraping/fox_sitemap.py                    # 全量抓所有 articles 分页（约 330 页，~10 分钟）
  python scraping/fox_sitemap.py --types articles videos --max-pages 50

每页结果存成 data/raw/fox/<type>_<page>.csv（url, lastmod, sitemap）；已存在的页自动跳过 → 可随时中断重跑。
时间过滤不在这里做（统一在 build_dataset.py 里按 lastmod 筛），这里尽量全收。
"""
from __future__ import annotations

import argparse
import html
import re
from urllib.parse import parse_qs, urlparse

from common import RAW_DIR, polite_get, write_rows

INDEX_URL = "https://www.foxnews.com/sitemap.xml"
OUT_DIR = RAW_DIR / "fox"
FIELDS = ["url", "lastmod", "sitemap"]

_BLOCK = re.compile(r"<(url|sitemap)>(.*?)</\1>", re.S | re.I)
_LOC = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.S | re.I)
_LASTMOD = re.compile(r"<lastmod>\s*(.*?)\s*</lastmod>", re.S | re.I)


def parse_sitemap(xml: str) -> list[dict]:
    """解析 sitemap 或 sitemap index，返回 [{loc, lastmod}]。用正则而非 XML 库：对命名空间/小格式错误更宽容。"""
    out = []
    for _, body in _BLOCK.findall(xml):
        loc = _LOC.search(body)
        if not loc:
            continue
        lm = _LASTMOD.search(body)
        out.append({"loc": html.unescape(loc.group(1)), "lastmod": lm.group(1) if lm else ""})
    return out


def child_sitemaps(index_xml: str, types: list[str]) -> list[tuple[str, str, int]]:
    """从索引里挑出指定 type 的子 sitemap，返回 [(url, type, page)]，按 page 排序。无 page 参数的记为 page=1。"""
    res = []
    for item in parse_sitemap(index_xml):
        q = parse_qs(urlparse(item["loc"]).query)
        t = (q.get("type") or [""])[0]
        if t in types:
            page = int((q.get("page") or ["1"])[0])
            res.append((item["loc"], t, page))
    return sorted(res, key=lambda x: (x[1], x[2]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["articles"], help="sitemap 类型，默认只要 articles")
    ap.add_argument("--max-pages", type=int, default=0, help="每种类型最多抓多少页，0=全部")
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--probe", action="store_true", help="只抓索引和第一个子 sitemap，打印样例")
    ap.add_argument("--start", default="2024-01",
                    help="分页按时间从新到旧；连续 2 页的最新 lastmod 都早于这个月份就停止")
    args = ap.parse_args()

    r = polite_get(INDEX_URL, delay=0)
    if r is None:
        raise SystemExit("拿不到 sitemap 索引，检查网络/是否被封")
    children = child_sitemaps(r.text, args.types)
    print(f"索引里找到 {len(children)} 个子 sitemap（types={args.types}）")
    if args.max_pages:
        children = [c for c in children if c[2] <= args.max_pages]

    if args.probe:
        url, t, page = children[0]
        rr = polite_get(url, delay=args.delay)
        items = parse_sitemap(rr.text) if rr else []
        print(f"[probe] {url} → {len(items)} 条")
        for it in items[:8]:
            print("   ", it["lastmod"], it["loc"])
        if items:
            lms = sorted(i["lastmod"] for i in items if i["lastmod"])
            print(f"[probe] 这一页 lastmod 范围: {lms[0] if lms else '?'} ~ {lms[-1] if lms else '?'}")
        return

    import csv

    total_new = 0
    old_streak: dict[str, int] = {}
    for i, (url, t, page) in enumerate(children, 1):
        if old_streak.get(t, 0) >= 2:
            continue
        out = OUT_DIR / f"{t}_{page:04d}.csv"
        if out.exists():
            with open(out) as f:
                lms = sorted(r["lastmod"] for r in csv.DictReader(f) if r["lastmod"])
        else:
            rr = polite_get(url, delay=args.delay)
            if rr is None:
                continue
            items = parse_sitemap(rr.text)
            rows = [{"url": it["loc"], "lastmod": it["lastmod"], "sitemap": f"{t}:{page}"} for it in items]
            write_rows(out, rows, FIELDS)
            total_new += len(rows)
            lms = sorted(x["lastmod"] for x in rows if x["lastmod"])
            rng = f"{lms[0][:10]}~{lms[-1][:10]}" if lms else "no-lastmod"
            print(f"[{i}/{len(children)}] {t} p{page}: {len(rows)} 条 ({rng})  累计新增 {total_new}", flush=True)
        # 这一页最新的日期都早于 --start → 计一次；连续两次就停止这个类型
        if lms and lms[-1][:7] < args.start:
            old_streak[t] = old_streak.get(t, 0) + 1
            if old_streak[t] >= 2:
                print(f"{t}: 连续 2 页早于 {args.start}，停止。")
        else:
            old_streak[t] = 0
    print(f"完成。文件在 {OUT_DIR}")


if __name__ == "__main__":
    main()
