from __future__ import annotations
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
PAD, UNK, BOS, EOS = ('<pad>', '<unk>', '<bos>', '<eos>')

@dataclass
class Token:
    form: str
    lemma: str
    upos: str
    feats: str
    # Dependency / misc columns, retained for round-trip CoNLL-U writing.
    # Defaults to "_" so every existing Token(form=, lemma=, upos=, feats=)
    # construction site keeps working; read_conllu populates them from gold.
    xpos: str = '_'
    head: str = '_'
    deprel: str = '_'
    deps: str = '_'
    misc: str = '_'

@dataclass
class Sentence:
    """A sentence.

    ``tokens`` holds the analytic tokens (one per ``ID`` column row that is a
    plain integer -- multiword ``n-m`` and empty-node ``n.m`` rows are
    intentionally excluded so they never enter the model). For round-trip
    CoNLL-U writing (cagf.predict_writer), the original ``comments`` and the
    verbatim ``misc_lines`` (multiword / empty-node rows) are preserved so the
    official conll18_ud_eval script can compare a regenerated file against the
    gold file token-for-token. Both fields default to empty, so every existing
    ``Sentence(tokens=...)`` construction site keeps working unchanged.
    """
    tokens: List[Token] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)
    misc_lines: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tokens)

def read_conllu(path: str | Path) -> List[Sentence]:
    sentences: List[Sentence] = []
    current_tokens: List[Token] = []
    current_comments: List[str] = []
    current_misc: List[str] = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip('\n')
            if not line.strip():
                if current_tokens:
                    sentences.append(Sentence(tokens=current_tokens, comments=current_comments, misc_lines=current_misc))
                    current_tokens, current_comments, current_misc = [], [], []
                continue
            if line.startswith('#'):
                current_comments.append(line)
                continue
            cols = line.split('\t')
            if len(cols) < 6:
                raise ValueError(f'{path}:{line_no}: expected >=6 tab-separated fields, got {len(cols)}')
            tid = cols[0]
            if '-' in tid or '.' in tid:
                # multiword token range (n-m) or empty node (n.m): keep verbatim
                # for round-trip CoNLL-U writing, but do not treat as a token.
                current_misc.append(line)
                continue
            form, lemma, upos, xpos, feats = (cols[1], cols[2], cols[3], cols[4], cols[5])
            # Pad short rows so HEAD/DEPREL/DEPS/MISC default to "_". The
            # official conll18_ud_eval script requires HEAD to be an integer,
            # and downstream metrics (even the morphology-only ones we report)
            # need the row to parse, so we must carry these through verbatim
            # from gold rather than writing "_" blindly.
            head = cols[6] if len(cols) > 6 and cols[6] else '_'
            deprel = cols[7] if len(cols) > 7 and cols[7] else '_'
            deps = cols[8] if len(cols) > 8 and cols[8] else '_'
            misc = cols[9] if len(cols) > 9 and cols[9] else '_'
            current_tokens.append(Token(form=form, lemma=lemma, upos=upos, feats=feats,
                                        xpos=xpos, head=head, deprel=deprel, deps=deps, misc=misc))
    if current_tokens:
        sentences.append(Sentence(tokens=current_tokens, comments=current_comments, misc_lines=current_misc))
    if not sentences:
        raise ValueError(f'{path}: no sentences were parsed. The file is either empty or not valid CoNLL-U -- check the source corpus before training.')
    return sentences

def _lcs_alignment(a: str, b: str) -> Tuple[int, int]:
    max_common = min(len(a), len(b))
    prefix = 0
    while prefix < max_common and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while suffix < max_common - prefix and a[len(a) - 1 - suffix] == b[len(b) - 1 - suffix]:
        suffix += 1
    return (prefix, suffix)

def form_to_edit_script(form: str, lemma: str) -> str:
    form_l, lemma_l = (form.lower(), lemma.lower())
    prefix, suffix = _lcs_alignment(form_l, lemma_l)
    deleted = form_l[prefix:len(form_l) - suffix] if suffix < len(form_l) - prefix else ''
    inserted = lemma_l[prefix:len(lemma_l) - suffix] if suffix < len(lemma_l) - prefix else ''
    return f'P{prefix}S{suffix}D{deleted}I{inserted}'

def apply_edit_script_verbose(form: str, script: str) -> Tuple[str, bool]:
    form_l = form.lower()
    try:
        rest = script[1:]
        p_str, rest = rest.split('S', 1)
        s_str, rest = rest.split('D', 1)
        deleted, inserted = rest.split('I', 1)
        prefix, suffix = (int(p_str), int(s_str))
        if prefix + suffix > len(form_l):
            return (form, False)
        head = form_l[:prefix]
        tail = form_l[len(form_l) - suffix:] if suffix > 0 else ''
        middle = form_l[prefix:len(form_l) - suffix]
        if middle != deleted:
            return (form, False)
        return (head + inserted + tail, True)
    except (ValueError, IndexError):
        return (form, False)


def restore_lemma_case(form: str, lemma: str, upos: str | None = None) -> str:
    """Recover the surface orthography (capitalisation) of ``form`` on ``lemma``.

    Edit-script rules operate on lower-cased form/lemma pairs (Bergmanis &
    Golding, 2018), so ``apply_edit_script`` returns a lower-case string even
    for proper nouns or acronyms whose gold lemma retains the original
    capitalisation. The official CoNLL-2018 scorer compares lemma strings
    *byte-for-byte*, so this loss of case costs several percentage points on
    lemma F1 for a Cyrillic-script corpus like UD_Kazakh-KTB in which a large
    fraction of tokens (akorda-random news, named entities) are capitalised.

    The recovery rule is deliberately conservative and UPOS-conditional so it
    does **not** over-correct sentence-initial common nouns (whose surface
    form is capitalised by virtue of sentence position, but whose lemma is
    lower-case):

      * all-caps surface form with length > 1 (``АҚШ``, ``КСРО``) → upper-case
        the lemma (acronyms keep their case);
      * otherwise, if the predicted UPOS is ``PROPN`` and the form starts with
        a capital → capitalise the lemma first character (proper nouns);
      * otherwise → leave the lemma lower-case (covers regular nouns, verbs,
        and sentence-initial common nouns).

    Passing ``upos=None`` disables the PROPN branch, restoring only all-caps
    acronyms; this is the safe default when UPOS is not yet predicted.
    """
    if not form or not lemma:
        return lemma
    # all-caps acronyms (length > 1 so a single capital like "I" does not
    # trigger an .upper() that happens to match, and so we don't title-case
    # regular initials).
    if form.isupper() and len(form) > 1:
        return lemma.upper()
    if upos == 'PROPN' and form[0].isupper():
        return lemma[0].upper() + lemma[1:]
    return lemma


def apply_edit_script(form: str, script: str, upos: str | None = None) -> str:
    """Apply an edit script to ``form`` and (optionally) restore surface case.

    ``upos``, when provided, enables UPOS-aware case recovery (see
    :func:`restore_lemma_case`). It is ``None`` by default for backward
    compatibility with callers that decode lemmas without a UPOS context
    (e.g. unit tests of the edit-script itself).
    """
    result, _valid = apply_edit_script_verbose(form, script)
    if upos is not None:
        result = restore_lemma_case(form, result, upos)
    return result

class Vocab:

    def __init__(self, itos: List[str]):
        self.itos = itos
        self.stoi = {s: i for i, s in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[UNK])

    def decode(self, idx: int) -> str:
        return self.itos[idx]

    @classmethod
    def build(cls, items: List[str], specials: List[str], min_freq: int=1, max_size: Optional[int]=None) -> 'Vocab':
        counts = Counter(items)
        vocab_items = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        return cls(itos=list(specials) + vocab_items)

    def to_json(self) -> Dict:
        return {'itos': self.itos}

    @classmethod
    def from_json(cls, obj: Dict) -> 'Vocab':
        return cls(itos=obj['itos'])

@dataclass
class CorpusVocabs:
    char_vocab: Vocab
    word_vocab: Vocab
    upos_vocab: Vocab
    grammeme_vocab: Vocab
    lemma_rule_vocab: Vocab

    def save(self, path: str | Path) -> None:
        obj = {'char_vocab': self.char_vocab.to_json(), 'word_vocab': self.word_vocab.to_json(), 'upos_vocab': self.upos_vocab.to_json(), 'grammeme_vocab': self.grammeme_vocab.to_json(), 'lemma_rule_vocab': self.lemma_rule_vocab.to_json()}
        Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path: str | Path) -> 'CorpusVocabs':
        obj = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(char_vocab=Vocab.from_json(obj['char_vocab']), word_vocab=Vocab.from_json(obj['word_vocab']), upos_vocab=Vocab.from_json(obj['upos_vocab']), grammeme_vocab=Vocab.from_json(obj['grammeme_vocab']), lemma_rule_vocab=Vocab.from_json(obj['lemma_rule_vocab']))

def parse_feats(feats: str) -> List[str]:
    if feats in ('', '_'):
        return []
    return feats.split('|')

def build_vocabs(train_sentences: List[Sentence], min_char_freq: int=1, min_word_freq: int=2, max_word_vocab: int=50000) -> CorpusVocabs:
    chars: List[str] = []
    words: List[str] = []
    upos_tags: List[str] = []
    grammemes: List[str] = []
    lemma_rules: List[str] = []
    for sent in train_sentences:
        for tok in sent.tokens:
            words.append(tok.form.lower())
            chars.extend(list(tok.form))
            upos_tags.append(tok.upos)
            grammemes.extend(parse_feats(tok.feats))
            lemma_rules.append(form_to_edit_script(tok.form, tok.lemma))
    char_vocab = Vocab.build(chars, specials=[PAD, UNK, BOS, EOS], min_freq=min_char_freq)
    word_vocab = Vocab.build(words, specials=[PAD, UNK], min_freq=min_word_freq, max_size=max_word_vocab)
    upos_vocab = Vocab.build(upos_tags, specials=[PAD, UNK])
    grammeme_vocab = Vocab.build(grammemes, specials=[UNK])
    lemma_rule_vocab = Vocab.build(lemma_rules, specials=[PAD, UNK])
    return CorpusVocabs(char_vocab=char_vocab, word_vocab=word_vocab, upos_vocab=upos_vocab, grammeme_vocab=grammeme_vocab, lemma_rule_vocab=lemma_rule_vocab)