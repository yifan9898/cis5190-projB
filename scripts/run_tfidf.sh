#!/usr/bin/env bash
# 训练 TF-IDF 线性模型 → 导出 submission/model.pt → 用官方 eval 在 val_split 上验证
# 用法（项目根目录）： bash scripts/run_tfidf.sh [额外参数传给 train_linear.py，如 --clf svm]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"

"$PY" experiments/train_linear.py --name tfidf_lr_official "$@"

echo "== 官方 eval（和排行榜同一套截断/加载/打分逻辑）=="
"$PY" template/eval_project_b.py \
    --model submission/model.py \
    --preprocess submission/preprocess.py \
    --csv data/val_split.csv \
    --weights submission/model.pt
