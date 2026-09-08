#!/bin/bash
# Task C (revision plan, R3.8): XLM-R base baseline under the exact same
# protocol as KazRoBERTa — same 6-config grid selected on fold-0 dev, same
# 10-fold stratified CV, same prediction targets and official scoring.
#
# Pinned revision e73636d4f797dec63c3081bb6ed5c7b0bb3f2089 (snapshot present
# in the local HF cache; 278M-param encoder). Folds at seed 42; gramprobs
# dumps per fold; jack-knife pooled scoring per config.
#
# ~28-30 h on MPS. Launch detached (survives terminal/app restarts):
#   nohup caffeinate -dims bash scripts/run_task_C_xlmr.sh >> logs/task_c_xlmr.log 2>&1 & disown
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
export PYTHONPATH=.
PY=.venv/bin/python

MN='FacebookAI/xlm-roberta-base'
REV='e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'
GRID_JSON='results_cv_xlmr/grid/grid_results.json'

echo "=== [C] XLM-R HP grid: 6 configs x 15 epochs, fold 0, dev-macro-F1 selection ==="
date
$PY scripts/kazroberta_hp_grid.py --model-name "$MN" --revision "$REV" \
    --epochs 15 --out "$GRID_JSON"

BEST_LR=$($PY -c "import json; b=max(json.load(open('$GRID_JSON'))['grid'], key=lambda r: r['dev_macro_f1']); print(b['lr'])")
BEST_BS=$($PY -c "import json; b=max(json.load(open('$GRID_JSON'))['grid'], key=lambda r: r['dev_macro_f1']); print(b['batch_size'])")
echo "=== [C] selected by dev fold 0: lr=$BEST_LR batch_size=$BEST_BS ==="
date

echo "=== [C] XLM-R gold-only, 10-fold stratified CV ==="
date
$PY scripts/run_cv_kazroberta.py \
    --model-name "$MN" --revision "$REV" \
    --configs gold_only --k 10 --strategy stratified --seed 42 \
    --max-epochs 80 --batch-size "$BEST_BS" --lr "$BEST_LR" \
    --out-dir results_cv_xlmr/stratified_gold

echo "=== [C] XLM-R silver-finetune, 10-fold stratified CV ==="
date
$PY scripts/run_cv_kazroberta.py \
    --model-name "$MN" --revision "$REV" \
    --configs silver_finetune --k 10 --strategy stratified --seed 42 \
    --max-epochs 80 --batch-size "$BEST_BS" --lr "$BEST_LR" \
    --silver-path data/silver/silver_unfiltered_subset_210k.conllu \
    --out-dir results_cv_xlmr/stratified_silver

echo "=== Task C complete ==="
date
