#!/usr/bin/env bash
# Overnight silver-scaling experiment: quality vs. silver-corpus size.
#
# Runs the silver-pretrain -> gold-finetune CV at four nested silver budgets
# (100K / 210K / 500K / 1M tokens), 10 folds each, on the full UD Kazakh-KTB
# (1078 sentences jack-knifed). Produces the scaling curve that answers the
# reviewer question "how much silver do you need, and where is saturation?".
#
# Subsets are NESTED (100K ⊂ 210K ⊂ 500K ⊂ 1M), generated once by
# scripts/make_silver_subset.py, so adjacent points on the curve differ only
# by added data -- not by redraw noise.
#
# Measured wall-clock on an RTX 4070 (full run, 2026-08-05): ~13 h total.
#   100K ~1.3h | 210K ~2.2h | 500K ~3.8h | 1M ~5.7h  (x10 folds each)
# Per-fold time is sublinear in corpus size: larger silver => earlier early
# stopping (best_epoch drops), so the last budget costs far less than a naive
# linear extrapolation would predict.
#
# Each budget writes to results_cv_silver/scaling_<budget>k/ and is safe to
# resume: --resume skips folds whose per_fold.json + fold_N.conllu already
# exist. Re-run this script after any interruption to continue.
#
# Usage (run from an active uv venv, or set PYTHON explicitly):
#   bash scripts/run_scaling_overnight.sh           # all 4 budgets, k=10
#   BUDGETS="100000 210000" bash scripts/run_scaling_overnight.sh   # subset
#   K=5 bash scripts/run_scaling_overnight.sh        # fewer folds for a pilot
#   PYTHON="$(which python)" bash scripts/run_scaling_overnight.sh
set -uo pipefail
cd "$(dirname "$0")/.."

# Interpreter: prefer $PYTHON, else the active venv's `python`, else `python3`.
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"
if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
  echo "ERROR: no usable python found. Activate your uv venv or set PYTHON=..." >&2
  exit 1
fi

BUDGETS="${BUDGETS:-100000 210000 500000 1000000}"
K="${K:-10}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1

mkdir -p logs
echo "=== silver-scaling overnight START $(date) ==="
echo "python=$PYTHON"
echo "budgets=[$BUDGETS] k=$K seed=$SEED device=$DEVICE"

# Track whether every budget succeeded; we run all of them even if one fails.
FAILED=0

for B in $BUDGETS; do
  SUFFIX="$((B / 1000))k"
  SILVER="data/silver/silver_subset_${SUFFIX}.conllu"
  OUT="results_cv_silver/scaling_${SUFFIX}"
  LOG="logs/scaling_${SUFFIX}.log"

  if [ ! -f "$SILVER" ]; then
    echo "ERROR: $SILVER not found. Run scripts/make_silver_subset.py first." >&2
    FAILED=1
    continue
  fi

  echo ""
  echo ">>> budget $B ($SUFFIX) -> $OUT   ($(date))"

  # NOTE: pipefail is set, so the python exit status propagates through tee.
  # We grep on the tee'd log AFTER python finishes instead of in the live
  # pipeline: that way `grep` returning "no matches" (e.g. python crashed
  # before printing any matching line) no longer aborts the whole script.
  if ! "$PYTHON" scripts/run_cv_silver.py \
        --silver-filtered "$SILVER" \
        --k "$K" --strategy stratified --seed "$SEED" \
        --configs silver_filtered_pretrain_ktb_finetune \
        --out-dir "$OUT" \
        --device "$DEVICE" --resume >"$LOG" 2>&1; then
    echo "ERROR: budget $SUFFIX failed -- see $LOG" >&2
    tail -n 20 "$LOG" >&2
    FAILED=1
    continue
  fi

  grep -E "fold [0-9]+ done|JACK-KNIFE|Corpus:|Folds:|resuming" "$LOG" || true
done

if [ "$FAILED" -ne 0 ]; then
  echo ""
  echo "=== silver-scaling overnight FINISHED WITH ERRORS $(date) ===" >&2
  exit 1
fi

echo ""
echo "=== silver-scaling overnight DONE $(date) ==="
echo "Aggregating curve..."
if ! "$PYTHON" scripts/scaling_curve.py \
      --budgets $BUDGETS \
      --root results_cv_silver \
      --json-out results_cv_silver/scaling_curve.json \
      --csv-out results_cv_silver/scaling_curve.csv; then
  echo "ERROR: scaling_curve.py failed" >&2
  exit 1
fi
