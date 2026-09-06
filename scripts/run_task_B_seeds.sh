#!/bin/bash
# Task B (revision plan, R3.6): seed replication, short schedule.
#
# CAGF gold-only + silver: seeds 13 and 2024 over the full 10-fold stratified
# protocol (the main paper claims rest on these; seed 42 already exists).
# KazRoBERTa: seeds 13/2024 on the PRE-FIXED folds {0,3,7} only -- enough to
# establish the run-to-run floor whose absence §3.6 concedes. Fold choice is
# fixed in advance and must not be changed after seeing results.
#
# Folds stay at seed 42 everywhere (--train-seed only varies initialization);
# silver-encoder pretraining is re-seeded per run (honest replicate).
# Grammeme probability dumps land next to each fold for the E.5 threshold sweep.
#
# ~30 h on MPS. Launch with:
#   caffeinate -dims bash scripts/run_task_B_seeds.sh 2>&1 | tee logs/task_b_seeds.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
export PYTHONPATH=.
PY=.venv/bin/python

for SEED in 13 2024; do
  echo "=== [B] CAGF gold-only + silver, seed $SEED, stratified k=10 ==="
  date
  $PY scripts/run_cv_silver.py \
    --k 10 --strategy stratified --seed 42 --train-seed "$SEED" \
    --configs ktb_only,silver_unfiltered_pretrain_ktb_finetune \
    --silver-unfiltered data/silver/silver_unfiltered_subset_210k.conllu \
    --out-dir "results_cv_silver/seed${SEED}"

  echo "=== [B] KazRoBERTa gold-only, seed $SEED, folds 0,3,7 ==="
  date
  $PY scripts/run_cv_kazroberta.py \
    --configs gold_only --k 10 --strategy stratified --seed 42 \
    --train-seed "$SEED" --folds 0,3,7 \
    --max-epochs 80 --batch-size 16 --lr 5e-5 \
    --out-dir "results_cv_kazroberta/seed${SEED}_gold"

  echo "=== [B] KazRoBERTa silver, seed $SEED, folds 0,3,7 ==="
  date
  $PY scripts/run_cv_kazroberta.py \
    --configs silver_finetune --k 10 --strategy stratified --seed 42 \
    --train-seed "$SEED" --folds 0,3,7 \
    --max-epochs 80 --batch-size 16 --lr 5e-5 \
    --silver-path data/silver/silver_unfiltered_subset_210k.conllu \
    --out-dir "results_cv_kazroberta/seed${SEED}_silver"
done

echo "=== Task B complete ==="
date
