#!/bin/bash
# P1a + A.7 LOSO (revision plan, R3.3 strictest protocol + R2.2 support).
#
# P1a: what the pure silver-pretrained model predicts on gold (fold 0) —
#      UPOS distribution / X share before any gold fine-tuning (~40 min).
# A.7: leave-one-source-out for Иран + wikipedia × 4 conditions, fixed
#      whole-document dev (~11%, seed 42); akorda-random/kdt are already
#      covered whole by the grouped-5 baskets of task A (~5.5-7 h).
#
# Launch detached:
#   nohup caffeinate -dims bash scripts/run_task_P1a_A7.sh >> logs/p1a_loso.log 2>&1 & disown
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
export PYTHONPATH=.
PY=.venv/bin/python

echo "=== [P1a] silver-pretrain-only probe (fold 0) ==="
date
$PY scripts/pre_finetune_probe.py

echo "=== [A.7] LOSO: Иран + wikipedia × {cagf,cazr} × {gold,silver} ==="
date
$PY scripts/run_loso.py \
    --silver data/silver/silver_unfiltered_subset_210k.conllu \
    --out-dir results_loso

echo "=== P1a + LOSO complete ==="
date
