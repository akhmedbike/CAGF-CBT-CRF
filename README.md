# CAGF-CBT+CRF — Character-Aware Gated Fusion CNN–BiLSTM–Transformer with CRF

Multi-task morphosyntactic analyzer for Kazakh: jointly predicts **lemma**,
**UPOS part-of-speech**, and **grammemes (UD features)** for the
Universal Dependencies Kazakh-KTB treebank.

The architecture combines a character-CNN encoder, a word-level BiLSTM, a
Transformer encoder, an adaptive gated-fusion layer, and a linear-chain CRF
for structured POS decoding. Lemmatization is formulated as **edit-script
classification** (not raw-lemma classification), which generalises to unseen
surface forms.

## Repository layout

```
cagf/              model, training loop, data, CRF, losses, baselines, device,
                   folds (CV), predict_writer (CoNLL-U), official_eval, feat_constraints
scripts/           experiment drivers (ablation, baselines, silver, CV, analysis)
configs/           default.yaml — all hyperparameters
tests/             92 pytest tests (unit + integration)
data/              gold UD + silver corpus. Gold is committed; the full silver
                   corpus and book skeleton are gitignored (see "Large files" below).
results/           ablation + baseline runs + generated tables (committed)
results_cv/        10-fold CV runs + official CoNLL-2018 metrics (committed JSON)
third_party/       vendored conll18_ud_eval.py (official UFAL scorer, MPL-2.0)
kaznlp/            [not in repo] clone nlacslab/kaznlp separately for silver
                   annotation — its CC-BY-SA 4.0 license is incompatible with MIT
docs/              [not in repo] internal working documents
models/            vocab files are committed; trained checkpoints are gitignored
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # torch 2.13 (MPS), numpy, scipy, sklearn, pyyaml

# KazNLP is needed only to (re)build the silver corpus. Clone separately:
git clone https://github.com/nlacslab/kaznlp.git
```

Python 3.14 on Apple Silicon is the reference environment (MPS accelerator).
On Linux/CUDA the same code runs unchanged (`cagf/device.py` auto-selects).

## Reproduce the paper's numbers

Everything is driven from `configs/default.yaml`. Results are written to JSON
and aggregated by the table scripts — every number in the manuscript is
regenerable from these artifacts.

```bash
# 1. Ablation study (5 configs x 5 seeds = 25 runs)  -> results/ablation_raw.json
PYTHONPATH=. .venv/bin/python scripts/run_ablation.py --config configs/default.yaml

# 2. Baselines (CNN / CNN-BiLSTM / CNN-BiLSTM-Transformer / subword-tagging)
PYTHONPATH=. .venv/bin/python scripts/train_baselines.py

# 3. Silver transfer learning (requires the silver corpus, step 5 below)
PYTHONPATH=.:kaznlp .venv/bin/python scripts/run_silver_ablation.py \
    --silver-filtered   data/silver/silver_subset_210k.conllu \
    --silver-unfiltered data/silver/silver_unfiltered_subset_210k.conllu \
    --seeds 13 42 123

# 4. Statistical analysis + tables
PYTHONPATH=. .venv/bin/python scripts/significance_test.py --holm --json-out results/significance.json
PYTHONPATH=. .venv/bin/python scripts/make_tables.py
PYTHONPATH=. .venv/bin/python scripts/efficiency.py
PYTHONPATH=. .venv/bin/python scripts/qualitative_crf.py

# 4b. Cross-validation (10-fold, official CoNLL-2018 metrics) -> results_cv/stratified/
#     This is the headline protocol: jack-knifed F1 over all 1078 sentences.
PYTHONPATH=. .venv/bin/python scripts/run_cv.py \
    --ablations full_model --k 10 --strategy stratified --seed 42 \
    --out-dir results_cv/stratified
PYTHONPATH=. .venv/bin/python scripts/count_impossible_feats.py \
    --gold data/gold_merged/gold_train.conllu \
    --pred results_cv/stratified/full_model/pred_all.conllu \
    --gold-test results_cv/stratified/full_model/gold_all.conllu

# 5. (Re)build the silver corpus from the connected book skeleton
PYTHONPATH=.:kaznlp .venv/bin/python scripts/build_silver_corpus.py \
    --backend kaznlp --kaznlp-model kaznlp/kaznlp/morphology/mdl
PYTHONPATH=. .venv/bin/python scripts/filter_silver_corpus.py
PYTHONPATH=. .venv/bin/python scripts/analyze_silver.py
```

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ --ignore=tests/test_webapp.py   # 92 tests
```

### Standalone Flask demo (`webapp/`)

`tests/test_webapp.py` covers the dual-backend demo. It auto-detects whichever
trained checkpoint is on disk and serves it: CAGF-CBT+CRF
(`models/interface_model.pt`, trained by `scripts/train_for_interface.py`) or
KazRoBERTa (`models/interface_hf_model.pt`, trained by
`scripts/train_hf_for_interface.py`, or exported from a CV run via
`run_cv_kazroberta.py --save-interface-checkpoint`). When both exist, CAGF is
preferred; move the CAGF checkpoint aside to serve KazRoBERTa.

```bash
.venv/bin/pip install Flask                            # demo-only dependency
PYTHONPATH=. .venv/bin/python scripts/train_hf_for_interface.py \
    --init-encoder results_cv_kazroberta/checkpoints/silver_encoder_kazroberta.pt
PYTHONPATH=. .venv/bin/flask --app webapp.app run      # http://127.0.0.1:5000
```

The webapp tests load the real model, so they self-skip until a checkpoint is
trained: `pytest tests/test_webapp.py` (CAGF) and `pytest tests/test_webapp_hf.py`
(HF, also `-m slow`). Run them after training the relevant checkpoint.

## Key results

### Cross-validation (headline protocol, 10-fold stratified, official CoNLL-2018)

Every sentence in UD Kazakh-KTB (1078) is scored exactly once, by the fold
whose test set held it. Metrics are computed by the official `conll18_ud_eval`
script (vendored in `third_party/`), so they are directly comparable to the UD
parsing literature.

| Configuration | Lemmas | UPOS | UFeats | AllTags |
|---|---|---|---|---|
| Transformer-only | 69.81 | 72.39 | 49.09 | 43.79 |
| w/o char encoder | 69.58 | 73.61 | 48.65 | 42.82 |
| w/o gated fusion | 71.61 | 82.75 | 62.15 | 56.57 |
| w/o CRF | 71.40 | 83.16 | 64.15 | 58.59 |
| **CAGF-CBT+CRF (full)** | **71.49** | 83.06 | **64.43** | **59.13** |

Jack-knifed F1 over all 1078 sentences, 10 folds. Per-fold std (n=10):
Lemmas 1.2pp, UPOS 2.0pp, UFeats 2.8pp. MPS noise floor ≈0.2pp (measured by
repeating wo_crf with identical config).

**Holm-corrected significance (n=10 folds, 16 hypotheses, all on identical code):**
- The **character encoder** and **word-level BiLSTM** are the load-bearing
  components — removing either drops UPOS by 9-12pp (p_Holm < 0.001, d ≈ 3).
  These are the standard Ma & Hovy (2016) components.
- **Gated fusion is NOT significant** after Holm correction (p_Holm 0.42-1.0;
  raw p 0.05-0.10 for UFeats/AllTags, point estimate +2.3pp UFeats, but does
  not survive correction against the 0.2pp noise floor).
- **CRF is neutral**: all p_Holm = 1.0, diffs within the noise floor. The
  earlier "CRF hurts" was a gradient-starvation artifact, now fixed
  (`docs/crf_diagnosis.md`).
- **Implication:** only the standard char-CNN+BiLSTM stack is statistically
  load-bearing at this corpus scale. The article's novelty must rest on
  silver-corpus transfer + parameter efficiency, not on gated fusion or CRF.

**Feature-impossibility masking** (post-hoc, no retraining): 0.99% of the
full model's predicted `(UPOS, Feature=Value)` combos were never seen in
training; masking them lifts UFeats F1 to **64.53** (+0.10pp) and AllTags to
**59.40** (+0.27pp).

### Silver transfer learning (single-split, gold KTB test set)

| Configuration | Lemma F1 | UPOS F1 | Grammeme F1 |
|---|---|---|---|
| Gold-only (from scratch) | 45.68 | 69.39 | 44.27 |
| **Silver-pretrain → gold-finetune** | **49.33** | **70.89** | **47.72** |

Silver pretraining on a 210K-token KazNLP-annotated subset of the 2.89M-token
book corpus gives a lemma-F1 gain. Note: at n=3 seeds the single-split
statistics are underpowered (Wilcoxon minimum p = 0.25); the CV protocol above
is the recommended basis for comparison. Full ablation + significance tables
are in `results/tables/`; CV results in `results/tables/cv_results.md`.

## Large files & external dependencies

Several artifacts are **not** in this git repository because they exceed
GitHub's size limits or are released under licenses incompatible with this
project's MIT license. Download them separately as described below.

### Large data artifacts & checkpoints

The following are available on **Zenodo** *(link to be added)* — download and
place them at the shown paths to reproduce the silver-corpus experiments and
the trained-model demos:

| Path | Size | Notes |
|---|---|---|
| `data/silver/silver.conllu` | 326 MB | Full silver corpus (KazNLP-annotated). |
| `data/silver/silver_filtered.conllu` | 326 MB | Filtered silver corpus. |
| `data/auxiliary/books_skeleton.conllu` | 127 MB | Connected book skeleton. Per-source redistribution rights must be verified (see `data/auxiliary/corpus_manifest.json`); released for research use. |
| `results_cv_kazroberta/checkpoints/silver_encoder_kazroberta.pt` | 319 MB | KazRoBERTa encoder checkpoint. |
| `models/interface_model.pt` | 10 MB | CAGF-CBT+CRF checkpoint for the `webapp/` demo. |

The 210K-token subsets (`data/silver/silver_subset_210k.conllu`,
`data/silver/silver_unfiltered_subset_210k.conllu`) and all gold/merged UD
data **are** committed, so the core experiments can be reproduced without the
Zenodo download.

### KazNLP (CC-BY-SA 4.0 — clone separately)

The [KazNLP](https://github.com/nlacslab/kaznlp) toolkit is required only to
(re)build the silver corpus. It is licensed under Creative Commons
Attribution-ShareAlike 4.0, a viral copyleft incompatible with this repo's MIT
license, so it is **not** vendored here. Clone it alongside this repo:

```bash
git clone https://github.com/nlacslab/kaznlp.git
```

## Data availability

- **UD Kazakh-KTB** (gold, in `data/gold/`, `data/gold_merged/`): public,
  https://universaldependencies.org/treebanks/kk_ktb/ — released under the
  Universal Dependencies data license (CC BY-SA 4.0).
- **Book corpus** (silver source): 33 Kazakh-language literary/non-fiction
  sources, ~2.89M tokens after segmentation. Available on reasonable request;
  per-source redistribution rights must be verified before public release
  (see `data/auxiliary/corpus_manifest.json`).
- **Silver corpus**: derived from the book corpus via KazNLP annotation; the
  full files are on Zenodo (see above), 210K-token subsets are committed.

## License

This project's source code is licensed under the **MIT License** (see
`LICENSE`). Third-party components retain their own licenses, documented in
`NOTICE`:

- `third_party/conll18_ud_eval.py` — Mozilla Public License 2.0 (© UFAL,
  Charles University; vendored verbatim, file remains under MPL-2.0).
- KazNLP — CC-BY-SA 4.0 (not included; clone separately).
- UD Kazakh-KTB data — CC BY-SA 4.0 (Universal Dependencies data license).
