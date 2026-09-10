#!/bin/bash
# Task D (revision plan, R3.8 extension): Stanza baseline, exact same folds.
#
# Waits for the P1a+A.7 queue to release the GPU, then trains Stanza POS
# (UPOS+FEATS) + lemmatizer per fold of the standard 10-fold stratified
# protocol: no tokenizer (pretokenized), no charlm, no pretrained vectors,
# model selection on each fold's dev, official conll18 scoring on test.
# ~2-2.5 h on MPS.
#
# Launch detached:
#   nohup caffeinate -dims bash scripts/run_task_D_stanza.sh >> logs/task_d_stanza.log 2>&1 & disown
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
export PYTHONPATH=.
PY=.venv/bin/python

echo "=== [D] waiting for P1a+A.7 queue to release the GPU ==="
date
while pgrep -f "run_task_P1a_A7.sh" > /dev/null; do sleep 120; done
echo "GPU free — starting Stanza CV"
date

$PY scripts/run_stanza_cv.py --out-dir results_stanza/stratified --device mps

echo "=== Task D complete ==="
date
