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
