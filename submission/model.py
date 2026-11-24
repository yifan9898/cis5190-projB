"""
提交用 model.py：哈希化的 词 1-2gram + 字符 2-5gram TF-IDF + slug 格式元特征 → 线性分类器，全部装在一个 torch nn.Module 里。

为什么这样设计
- 后端只会把 model.pt 当 state_dict 加载，不能 pickle sklearn 对象。
- 用哈希代替词表：特征 id = crc32("w:" + gram) % NUM_BUCKETS，无需存字符串，跨平台确定。
- idf 存成 buffer；线性层用 EmbeddingBag(mode='sum', per_sample_weights=tfidf 值)，
  等价于稀疏向量 · 权重矩阵，CPU 推理很快。
- 训练在 sklearn 里做（experiments/train_linear.py），再把系数拷进来。特征提取函数两边共用这一份。

输入：preprocess.py 给出的小写空格分隔文本。输出：'FoxNews' / 'NBC' 字符串列表。
"""
from __future__ import annotations

import math
import zlib
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn

NUM_BUCKETS = 2**19
WORD_NGRAMS = (1, 2)
CHAR_NGRAMS = (2, 5)
LABELS = ["FoxNews", "NBC"]  # 输出类别顺序：index 1 = NBC
META_SCALE = 1.0  # 元特征组（长度/首尾词等）L2 归一化后再乘的系数


def _h(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) % NUM_BUCKETS


def meta_tokens(words: List[str]) -> List[str]:
    """slug 的格式特征（5 折 CV 上 +0.9 个点）：
    词数、字符长度、最后一个词的长度（NBC slug 常在固定长度截断，末词是半截）、首词、末词、
    是否出现孤立的 's'（NBC 把撇号转成 '-s-'，如 trump-s；Fox 一般写成 trumps）。"""
    if not words:
        return ["len=0"]
    text_len = len(" ".join(words))
    toks = [f"len={min(len(words), 25)}", f"charlen={min(text_len // 10, 12)}",
            f"lastlen={min(len(words[-1]), 6)}", "first=" + words[0], "last=" + words[-1]]
    if "s" in words:
        toks.append("tok_s")
    return toks


def raw_features(text: str) -> List[Tuple[Dict[int, float], str]]:
    """返回三组 (哈希id → 原始计数)：词 n-gram、字符 n-gram、元特征。每组分别 L2 归一化（同 sklearn FeatureUnion）。"""
    words = text.lower().split()
    wc: Counter = Counter()
    for n in range(WORD_NGRAMS[0], WORD_NGRAMS[1] + 1):
        for i in range(len(words) - n + 1):
            wc[_h("w:" + " ".join(words[i : i + n]))] += 1
    cc: Counter = Counter()
    for w in words:  # 同 sklearn analyzer='char_wb'：每个词两侧补空格
        p = f" {w} "
        for n in range(CHAR_NGRAMS[0], CHAR_NGRAMS[1] + 1):
            for i in range(len(p) - n + 1):
                cc[_h("c:" + p[i : i + n])] += 1
    mc: Counter = Counter(_h("m:" + t) for t in meta_tokens(words))
    return [(dict(wc), "w"), (dict(cc), "c"), (dict(mc), "m")]


def tfidf_vector(text: str, idf: torch.Tensor | List[float]) -> Tuple[List[int], List[float]]:
    """词/字符组：sublinear tf (1+log tf) × idf；元特征组：计数本身（不乘 idf）× META_SCALE。
    每组分别 L2 归一化。哈希碰撞的特征在 EmbeddingBag(sum) 里直接相加（与训练端 csr 求和一致）。"""
    ids: List[int] = []
    vals: List[float] = []
    for counts, kind in raw_features(text):
        if not counts:
            continue
        g_ids = list(counts.keys())
        if kind == "m":
            g_vals = [float(counts[j]) for j in g_ids]
        else:
            g_vals = [(1.0 + math.log(counts[j])) * float(idf[j]) for j in g_ids]
        norm = math.sqrt(sum(v * v for v in g_vals)) or 1.0
        scale = META_SCALE if kind == "m" else 1.0
        ids.extend(g_ids)
        vals.extend(scale * v / norm for v in g_vals)
    return ids, vals


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.EmbeddingBag(NUM_BUCKETS, 2, mode="sum")
        self.bias = nn.Parameter(torch.zeros(2))
        self.register_buffer("idf", torch.ones(NUM_BUCKETS))
        with torch.no_grad():
            self.weight.weight.zero_()

    def logits(self, texts: Iterable[str]) -> torch.Tensor:
        idf = self.idf.tolist()  # 一次性转 list，避免逐元素访问 tensor 的开销
        all_ids: List[int] = []
        all_vals: List[float] = []
        offsets: List[int] = []
        for t in texts:
            offsets.append(len(all_ids))
            ids, vals = tfidf_vector(str(t), idf)
            if not ids:  # 空输入：用 0 号桶、权重 0 占位
                ids, vals = [0], [0.0]
            all_ids.extend(ids)
            all_vals.extend(vals)
        out = self.weight(
            torch.tensor(all_ids, dtype=torch.long),
            torch.tensor(offsets, dtype=torch.long),
            per_sample_weights=torch.tensor(all_vals, dtype=torch.float32),
        )
        return out + self.bias

    def forward(self, texts: Iterable[str]) -> torch.Tensor:
        return self.logits(texts)

    @torch.no_grad()
    def predict(self, batch: Iterable[str]) -> List[str]:
        idx = self.logits(list(batch)).argmax(dim=-1).tolist()
        return [LABELS[i] for i in idx]


def get_model() -> Model:
    return Model()
