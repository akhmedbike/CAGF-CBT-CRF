# Vendored third-party code

## `conll18_ud_eval.py`

**Source:** official Universal Dependencies evaluation script, downloaded from
`https://universaldependencies.org/conll18/conll18_ud_eval.py`.

**Authors:** Milan Straka, Martin Popel — Institute of Formal and Applied
Linguistics (UFAL), Faculty of Mathematics and Physics, Charles University,
Czech Republic.

**License:** Mozilla Public License 2.0 (full text at
https://www.mozilla.org/MPL/2.0/).

**Version:** 1.1 (2 May 2018) — the version used for the official CoNLL 2018
shared task on multilingual parsing. The changelog is at the top of the file.

**Why it is here:** this is the canonical script for computing the F1 metrics
reported in UD parsing papers (`Lemmas`, `UPOS`, `UFeats`, `AllTags`). It is
vendored verbatim — **not modified, not adapted** — so that every number we
report is reproducible with the exact same code the UD community uses. The
only consumer is `cagf/official_eval.py`, which imports it as a module and
extracts the four F1 metrics.

**How to update:** re-download from the URL above, replace this file, and
re-run `tests/test_official_eval.py` — the gold-vs-gold test must still return
exactly 100.0 on every metric.

## UD treebank snapshots (`UD_Kazakh-KTB/`, `UD_Kyrgyz-KTMU/`, `UD_Turkish-IMST/`)

**Source:** official Universal Dependencies GitHub repositories
(`github.com/UniversalDependencies/<name>`), snapshotted with their nested
`.git/` directories removed. Each directory retains the upstream `LICENSE.txt`,
`README.md`, and `stats.xml`.

**License:** Creative Commons Attribution-ShareAlike 4.0 International — the
standard Universal Dependencies data license. It applies to the data files
only, not to this repository's code.

**Why they are here:** frozen inputs of the cross-lingual transfer
experiments (`results_transfer/`). Kazakh-KTB is the gold treebank whose
train/test splits feed `data/gold_merged/` via `scripts/merge_gold_corpus.py`;
the Kyrgyz-KTMU and Turkish-IMST treebanks are the related-language sources for
the transfer ablations. The committed snapshots are the exact versions used
for the published numbers.

**How to update:** re-clone the treebank from UD GitHub, remove its nested
`.git/` directory, and re-run the affected drivers.
