"""
训练 哈希 TF-IDF + 线性分类器，导出成可提交的 submission/model.pt，并记录实验结果。

用法（项目根目录，激活 .venv）：
  python experiments/train_linear.py --name tfidf_lr_official
  python experiments/train_linear.py --clf svm --C 0.5 --name tfidf_svm_official
  python experiments/train_linear.py --extra data/scraped_clean.csv --official-weight 20 --section-reweight --name tfidf_lr_rw

评测协议（所有实验统一）：
  - 验证集固定为 data/val_split.csv（官方数据切出的 960 条，任何训练数据都不得包含）
  - 输入一律先经过 reduce_url（和排行榜一致）
  - 结果追加到 experiments/results.csv（报告里的折线图从这里画）
脚本最后会用 torch 模型重新预测验证集，检查与 sklearn 预测一致，确保导出无误。
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
import model as M  # noqa: E402
from preprocess import prepare_data  # noqa: E402

RESULTS = ROOT / "experiments" / "results.csv"


def load(paths: list[str]) -> tuple[list[str], np.ndarray, list[str]]:
    """返回 (文本, 标签 0=Fox/1=NBC, 原始 url)。url 只用来算栏目做加权，不进模型。"""
    X, y, U = [], [], []
    for p in paths:
        fp = str(ROOT / p) if not Path(p).is_absolute() else p
        xs, ys = prepare_data(fp)
        with open(fp, newline="") as f:
            urls = [r[0] for r in csv.reader(f) if r]
        if urls and urls[0].strip().lower() == "url":
            urls = urls[1:]
        assert len(urls) == len(xs), f"{p}: url 行数与 prepare_data 输出不一致"
        for a, b, u in zip(xs, ys, urls):
            if b:  # 训练/验证数据都应有标签；没有的（不是这两家）丢掉
                X.append(a)
                y.append(1 if b == "NBC" else 0)
                U.append(u)
    return X, np.array(y), U


def section_of(url: str) -> str:
    segs = [s for s in urlparse(url.strip()).path.split("/") if s]
    return segs[0] if len(segs) > 1 else ""


def sample_weights(y_off, u_off, y_ext, u_ext, official_weight: float, section_reweight: bool, eps: float = 0.02):
    """官方样本权重 = official_weight。
    额外（爬取）样本：若 section_reweight，则按 "官方栏目占比 / 爬取栏目占比" 重新加权（分标签计算），
    让爬取数据的栏目分布贴近官方（也就是测试集）分布；官方没有的栏目给 eps 的小份额。
    额外样本的权重总和保持等于其样本数。"""
    w_off = np.full(len(y_off), official_weight, dtype=np.float64)
    w_ext = np.ones(len(y_ext), dtype=np.float64)
    if section_reweight and len(y_ext):
        s_off = np.array([section_of(u) for u in u_off])
        s_ext = np.array([section_of(u) for u in u_ext])
        for lab in (0, 1):
            mo, me = y_off == lab, y_ext == lab
            if not me.any():
                continue
            o = Counter(s_off[mo]); e = Counter(s_ext[me])
            no, ne = sum(o.values()), sum(e.values())
            tgt = {k: o.get(k, 0) / max(no, 1) + eps for k in e}
            z = sum(tgt.values())
            ratio = {k: (tgt[k] / z) / (e[k] / ne) for k in e}
            w_ext[me] = [ratio[k] for k in s_ext[me]]
            w_ext[me] *= me.sum() / w_ext[me].sum()
    return w_off, w_ext


def fit_idf(texts: list[str]) -> np.ndarray:
    df: Counter = Counter()
    for t in texts:
        seen = set()
        for counts, _ in M.raw_features(t):
            seen.update(counts.keys())
        df.update(seen)
    n = len(texts)
    idf = np.full(M.NUM_BUCKETS, math.log((1 + n) / 1) + 1.0, dtype=np.float64)
    for j, d in df.items():
        idf[j] = math.log((1 + n) / (1 + d)) + 1.0  # 同 sklearn smooth_idf
    return idf.astype(np.float32)


def vectorize(texts: list[str], idf: np.ndarray) -> csr_matrix:
    idf_list = idf.tolist()
    indptr, indices, data = [0], [], []
    for t in texts:
        ids, vals = M.tfidf_vector(t, idf_list)
        indices.extend(ids)
        data.extend(vals)
        indptr.append(len(indices))
    # 同一哈希桶的重复项在 csr 里会被求和，和 EmbeddingBag(sum) 行为一致
    return csr_matrix((data, indices, indptr), shape=(len(texts), M.NUM_BUCKETS), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["data/train_split.csv"], help="官方（目标分布）训练数据")
    ap.add_argument("--extra", nargs="*", default=[], help="额外数据，如 data/scraped_clean.csv（只读第一列 url）")
    ap.add_argument("--official-weight", type=float, default=5.0, help="官方样本的权重倍数")
    ap.add_argument("--section-reweight", action="store_true", help="按栏目把额外数据重新加权到官方分布")
    ap.add_argument("--val", default="data/val_split.csv")
    ap.add_argument("--clf", choices=["lr", "svm"], default="lr")
    ap.add_argument("--C", type=float, default=None, help="默认 lr=3, svm=0.3（5 折 CV 选出）")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="submission/model.pt")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    C = args.C if args.C is not None else (3.0 if args.clf == "lr" else 0.3)
    name = args.name or f"tfidf_{args.clf}_C{C}"

    t0 = time.time()
    Xo, yo, Uo = load(args.train)
    Xe, ye, Ue = load(args.extra) if args.extra else ([], np.array([], dtype=int), [])
    wo, we = sample_weights(yo, Uo, ye, Ue, args.official_weight, args.section_reweight)
    Xtr = Xo + Xe
    ytr = np.concatenate([yo, ye]).astype(int)
    wtr = np.concatenate([wo, we])
    Xva, yva, _ = load([args.val])
    print(f"官方 {len(Xo)} 条 ×{args.official_weight} | 额外 {len(Xe)} 条"
          f"{'（按栏目重加权）' if args.section_reweight else ''}")
    # 防泄露检查：验证集文本不得出现在训练集中
    overlap = len(set(Xva) & set(Xtr))
    print(f"train {len(Xtr)} (NBC {ytr.mean():.1%}) | val {len(Xva)} | 验证集文本与训练集完全重复: {overlap}")

    idf = fit_idf(Xtr)
    Atr, Ava = vectorize(Xtr, idf), vectorize(Xva, idf)
    print(f"特征提取 {time.time() - t0:.1f}s, 非零特征/样本 ≈ {Atr.nnz / max(1, Atr.shape[0]):.0f}")

    if args.clf == "lr":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=C, max_iter=5000)
    else:
        from sklearn.svm import LinearSVC
        clf = LinearSVC(C=C)
    clf.fit(Atr, ytr, sample_weight=wtr)
    tr_acc = float((clf.predict(Atr) == ytr).mean())
    sk_pred = clf.predict(Ava)
    va_acc = float((sk_pred == yva).mean())
    se = math.sqrt(va_acc * (1 - va_acc) / len(yva))
    print(f"[{name}] train acc {tr_acc:.4f} | val acc {va_acc:.4f} ± {se:.4f} (1 s.e.) | {time.time() - t0:.1f}s")

    # 导出成 torch 模型：class1 (NBC) 的 logit = w·x + b，class0 logit = 0
    import torch
    model = M.Model()
    with torch.no_grad():
        model.weight.weight[:, 1] = torch.from_numpy(clf.coef_[0].astype(np.float32))
        model.bias[1] = float(clf.intercept_[0])
        model.idf.copy_(torch.from_numpy(idf))
    model.eval()
    t1 = time.time()
    torch_pred = []
    for i in range(0, len(Xva), 32):
        torch_pred += model.predict(Xva[i : i + 32])
    torch_pred = np.array([1 if p == "NBC" else 0 for p in torch_pred])
    agree = float((torch_pred == sk_pred).mean())
    print(f"torch 导出自检：与 sklearn 预测一致率 {agree:.4f}，torch 推理 {1000 * (time.time() - t1) / len(Xva):.3f} ms/条")
    if agree < 0.995:
        raise SystemExit("导出不一致！不要提交这个 model.pt")

    if not args.no_save:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out)
        print(f"已保存 {out} ({out.stat().st_size / 1e6:.1f} MB)")

    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "name", "clf", "C", "train_files", "extra_files", "official_weight",
                        "section_reweight", "n_train", "train_acc", "val_acc", "val_se"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), name, args.clf, C, "+".join(args.train),
                    "+".join(args.extra), args.official_weight, args.section_reweight,
                    len(Xtr), round(tr_acc, 4), round(va_acc, 4), round(se, 4)])


if __name__ == "__main__":
    main()
