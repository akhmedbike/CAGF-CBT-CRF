from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
BACK_VOWELS = set('аоұыАОҰЫ')
FRONT_VOWELS = set('әеіөүӘЕІӨҮ')
ALL_VOWELS = BACK_VOWELS | FRONT_VOWELS

def last_vowel(word: str) -> Optional[str]:
    for ch in reversed(word):
        if ch in ALL_VOWELS:
            return ch
    return None

def harmony_class(word: str) -> Optional[str]:
    v = last_vowel(word)
    if v is None:
        return None
    return 'back' if v in BACK_VOWELS else 'front'
PLURAL_SUFFIXES = {'back': ['лар', 'дар', 'тар'], 'front': ['лер', 'дер', 'тер']}
CASE_SUFFIXES = {'Gen': {'back': ['ның', 'дың', 'тың'], 'front': ['нің', 'дің', 'тің']}, 'Acc': {'back': ['ны', 'ды', 'ты'], 'front': ['ні', 'ді', 'ті']}, 'Dat': {'back': ['ға', 'қа'], 'front': ['ге', 'ке']}, 'Loc': {'back': ['да', 'та'], 'front': ['де', 'те']}, 'Abl': {'back': ['дан', 'тан'], 'front': ['ден', 'тен']}, 'Ins': {'back': ['мен', 'бен', 'пен'], 'front': ['мен', 'бен', 'пен']}}
CASE_ORDER = ['Abl', 'Gen', 'Ins', 'Dat', 'Loc', 'Acc']

def _strip_longest_match(word: str, candidates: list[str]) -> Optional[str]:
    best = None
    for suf in candidates:
        if word.lower().endswith(suf) and len(word) - len(suf) >= 2:
            if best is None or len(suf) > len(best):
                best = suf
    return best

@dataclass
class MorphGuess:
    lemma_guess: str
    number: Optional[str]
    case: Optional[str]
    matched_suffix: str
    confidence: str

def guess_nominal_morphology(word: str) -> MorphGuess:
    hc = harmony_class(word)
    if hc is None:
        return MorphGuess(lemma_guess=word, number=None, case=None, matched_suffix='', confidence='low')
    stem = word
    matched_parts: list[str] = []
    matched_case = None
    best_case_suffix = None
    for case_name in CASE_ORDER:
        candidates = CASE_SUFFIXES[case_name][hc]
        suf = _strip_longest_match(stem, candidates)
        if suf is not None and (best_case_suffix is None or len(suf) > len(best_case_suffix)):
            matched_case = case_name
            best_case_suffix = suf
    if matched_case is not None:
        stem = stem[:-len(best_case_suffix)]
        matched_parts.append(best_case_suffix)
    hc_stem = harmony_class(stem) or hc
    plur_candidates = PLURAL_SUFFIXES[hc_stem]
    plur_suf = _strip_longest_match(stem, plur_candidates)
    matched_plural = False
    if plur_suf is not None:
        stem = stem[:-len(plur_suf)]
        matched_parts.append(plur_suf)
        matched_plural = True
    if not matched_parts:
        return MorphGuess(lemma_guess=word, number=None, case=None, matched_suffix='', confidence='low')
    confidence = 'high' if len(stem) >= 2 else 'low'
    return MorphGuess(lemma_guess=stem, number='Plur' if matched_plural else None, case=matched_case, matched_suffix='+'.join(reversed(matched_parts)), confidence=confidence)