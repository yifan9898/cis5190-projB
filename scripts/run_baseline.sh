#!/usr/bin/env bash
# 跑通 baseline：在 train_split 上重新训练，再用官方 eval 在 val_split 上评测
# 用法（在项目根目录 cis519-projB 下）：  bash scripts/run_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"

mkdir -p runs/baseline
echo "== 1/2 训练 baseline（train_split.csv, 3840 条）=="
# train_baseline.py 会把 model.pt 存到当前目录，所以 cd 到 runs/baseline 再跑
(cd runs/baseline && "../../$PY" ../../template/tested_baseline/train_baseline.py \
    --csv ../../data/train_split.csv --epochs 5)

echo "== 2/2 官方评测脚本（val_split.csv, 960 条，链接会被截成只剩标题词）=="
"$PY" template/eval_project_b.py \
    --model template/tested_baseline/model.py \
    --preprocess template/tested_baseline/preprocess.py \
    --csv data/val_split.csv \
    --weights runs/baseline/model.pt

# 可选：如果你从 Drive 下载了官方 model.pt 放到 template/tested_baseline/ 下，
# 取消下面几行注释也评一下（注意官方权重可能见过 val 里的样本，分数会偏高）
# "$PY" template/eval_project_b.py \
#     --model template/tested_baseline/model.py \
#     --preprocess template/tested_baseline/preprocess.py \
#     --csv data/val_split.csv \
#     --weights template/tested_baseline/model.pt
