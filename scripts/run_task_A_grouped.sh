#!/bin/bash
# Task A (revision applsci-4523162, R3.3 + R2.3): source-grouped 5-fold CV.
#
# Fold layout comes from cagf/folds.py grouped mode (LPT bin-packing of whole
# documents, dev = next basket): test sizes [318, 288, 157, 158, 157], baskets
# 0/1 = akorda-random / kdt whole. Hyperparameters are IDENTICAL to the
# stratified runs (CAGF: config defaults; KazRoBERTa: 80 epochs, batch 16,
# lr 5e-5) -- no new grid under grouping, per revision plan.
#
# Sequential MPS queue, ~12h total. Launch with:
#   caffeinate -dims bash scripts/run_task_A_grouped.sh 2>&1 | tee logs/task_a_grouped.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
export PYTHONPATH=.
PY=.venv/bin/python

echo "=== [A1/A2] CAGF gold-only + silver->gold, grouped k=5 ==="
date
$PY scripts/run_cv_silver.py \
  --k 5 --strategy grouped --seed 42 \
  --configs ktb_only,silver_unfiltered_pretrain_ktb_finetune \
  --silver-unfiltered data/silver/silver_unfiltered_subset_210k.conllu \
  --out-dir results_cv_silver/grouped5

echo "=== [A3] KazRoBERTa gold-only, grouped k=5 ==="
date
$PY scripts/run_cv_kazroberta.py \
  --configs gold_only --k 5 --strategy grouped --seed 42 \
  --max-epochs 80 --batch-size 16 --lr 5e-5 \
  --out-dir results_cv_kazroberta/grouped5

echo "=== [A4] KazRoBERTa silver->gold, grouped k=5 (one-time encoder pretrain included) ==="
date
$PY scripts/run_cv_kazroberta.py \
  --configs silver_finetune --k 5 --strategy grouped --seed 42 \
  --max-epochs 80 --batch-size 16 --lr 5e-5 \
  --silver-path data/silver/silver_unfiltered_subset_210k.conllu \
  --out-dir results_cv_kazroberta/grouped5

echo "=== Task A complete ==="
date
