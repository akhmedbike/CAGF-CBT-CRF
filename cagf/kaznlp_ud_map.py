"""Deterministic mapping from the KazNLP (KLC) morphological tag set to the
Universal Dependencies label space used by UD_Kazakh-KTB.

The mapping is derived from two authoritative sources:

  * the official KazNLP tag-set glossary
    (kaznlp/docs/morphology-tagset.MD), which defines the KLC POS tags
    (``R_ZE``, ``R_ET`` ...) and the grammatical-morpheme tags
    (``C4``, ``S3``, ``M4`` ...); and
  * the actual UPOS and FEATS inventory of the gold corpus
    (``results/vocabs.json``): 17 UPOS categories and 54 grammeme values.

Design decisions (all deliberate and documented, not incidental):

1.  Every FEATS value produced by this module is *gated* against the gold
    grammeme inventory (:data:`GOLD_GRAMMEMES`). Anything that would fall
    outside that inventory is dropped and counted. This guarantees that the
    silver label space is a subset of the gold label space, which is what
    makes silver-pretraining -> gold-fine-tuning label-compatible.

2.  A number of KLC morphemes are *derivational* rather than inflectional
    (``S9`` -нікі, ``LATT`` -дағы, ``SML`` -дай, ``ABE`` -сыз, ``EQU`` -ша)
    or have no counterpart in the gold FEATS inventory (``V4`` causative).
    These are intentionally left unmapped; see :data:`UNMAPPED_MORPHEMES`.

3.  Four unmarked-category defaults are applied as heuristics, matching UD
    conventions and the gold annotation style: nominative case and singular
    number on nominals with no overt case/number morpheme, and
    ``VerbForm=Fin|Mood=Ind`` on finite verbs with no explicit non-finite
    or non-indicative morpheme. These are marked HEURISTIC below and should
    be validated by a Kazakh linguist before the silver corpus is treated
    as anything other than an auxiliary, machine-annotated resource.

Because both the KLC->UD table and the disambiguated KazNLP analysis are
deterministic, the whole silver-corpus construction is reproducible given a
fixed KazNLP model version and this module.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Bumped whenever the tables below change; written into the silver corpus
# provenance header so a corpus can be tied to the exact mapping that made it.
MAPPING_VERSION = "1.0.0"

# The 54 gold grammeme values (from results/vocabs.json, minus <unk>). Used to
# gate every FEATS value this module emits.
GOLD_GRAMMEMES = frozenset({
    "Aspect=Hab", "Aspect=Imp", "Aspect=Perf",
    "Case=Abl", "Case=Acc", "Case=Dat", "Case=Gen", "Case=Ins", "Case=Loc", "Case=Nom",
    "Degree=Cmp", "Evident=Fh", "Gender=Fem", "Gender=Masc",
    "Mood=Cnd", "Mood=Des", "Mood=Imp", "Mood=Ind", "Mood=Opt", "Mood=Pot",
    "NumType=Card", "NumType=Card,Ord", "NumType=Coll", "NumType=Ord",
    "Number=Plur", "Number=Sing",
    "Number[psor]=Plur", "Number[psor]=Plur,Sing", "Number[psor]=Sing",
    "Person=1", "Person=2", "Person=3",
    "Person[psor]=1", "Person[psor]=2", "Person[psor]=3",
    "Polarity=Neg", "Polite=Form",
    "PronType=Dem", "PronType=Ind", "PronType=Int", "PronType=Neg", "PronType=Prs", "PronType=Tot",
    "Reflex=Yes",
    "Tense=Fut", "Tense=Past", "Tense=Pres",
    "VerbForm=Conv", "VerbForm=Fin", "VerbForm=Ger", "VerbForm=Inf", "VerbForm=Part",
    "Voice=Pass", "Voice=Rcp",
})

# ---------------------------------------------------------------------------
# POS: KLC root tag -> UD UPOS
# ---------------------------------------------------------------------------
POS_MAP: Dict[str, str] = {
    "R_ZE": "NOUN",
    "R_ZEQ": "PROPN",
    "R_ET": "VERB",
    "R_ETK": "AUX",
    "R_ETP": "VERB",
    "R_ETPK": "AUX",
    "R_ETB": "AUX",     # жоқ / емес used as a negator behaves like an auxiliary in KTB
    "R_SE": "ADJ",
    "R_SIM": "PRON",    # KLC conflates pronouns and determiners; PRON is the majority class
    "R_US": "ADV",
    "R_ZHL": "CCONJ",   # KLC conflates coordinating and subordinating conjunctions
    "R_SN": "NUM",
    "R_SH": "ADP",      # KLC conflates adpositions and particles; ADP is the default
    "R_MOD": "PART",
    "R_OS": "INTJ",
    "R_ELK": "INTJ",    # imitatives / onomatopoeia
    "R_SYM": "SYM",
    "R_BOS": "X",       # foreign word
    "R_X": "X",         # not analysed
}
# every punctuation root tag maps to PUNCT
_PUNCT_TAGS = {
    "R_NKT", "R_UTR", "R_DPH", "R_ATRN", "R_ZTRN", "R_TRN", "R_QNKT", "R_SUR",
    "R_AZZ", "R_ZZZ", "R_LEP", "R_UNKT", "R_SLH", "R_APS", "R_BSLH",
}
for _t in _PUNCT_TAGS:
    POS_MAP[_t] = "PUNCT"

_NOMINAL_UPOS = {"NOUN", "PROPN", "PRON", "ADJ", "NUM"}
_VERBAL_UPOS = {"VERB", "AUX"}

# ---------------------------------------------------------------------------
# Grammatical morphemes -> canonical UD FEATS (later gated by GOLD_GRAMMEMES)
# Each entry is a list of "Attr=Val" strings.
# ---------------------------------------------------------------------------
MORPH_MAP: Dict[str, List[str]] = {
    # number
    "N1": ["Number=Plur"],
    "N1S": ["Number=Plur"],
    # possessive (Number[psor]/Person[psor])
    "S1": ["Number[psor]=Sing", "Person[psor]=1"],
    "S2": ["Number[psor]=Sing", "Person[psor]=2"],
    "S3": ["Person[psor]=3"],
    "S3SIM": ["Person[psor]=3"],
    "S4": ["Number[psor]=Sing", "Person[psor]=2", "Polite=Form"],
    "S5": ["Number[psor]=Plur", "Person[psor]=1"],
    # case
    "C2": ["Case=Gen"],
    "C3": ["Case=Dat"],
    "C4": ["Case=Acc"],
    "C5": ["Case=Loc"],
    "C6": ["Case=Abl"],
    "C7": ["Case=Ins"],
    # degree
    "CMP": ["Degree=Cmp"],
    # voice
    "V1": ["Reflex=Yes"],
    "V2": ["Voice=Pass"],
    "V3": ["Voice=Rcp"],
    # mood
    "M2": ["Mood=Imp"],
    "M3": ["Mood=Des"],
    "M4": ["Mood=Cnd"],
    # tense (aorist T1 is analysed as present; T2 future; T3/T3E past)
    "T1": ["Tense=Pres"],
    "T2": ["Tense=Fut"],
    "T3": ["Tense=Past"],
    "T3E": ["Tense=Past"],
    # subject agreement (Number/Person)
    "P1": ["Number=Sing", "Person=1"],
    "P2": ["Number=Sing", "Person=2"],
    "P3": ["Person=3"],
    "P4": ["Number=Sing", "Person=2", "Polite=Form"],
    "P5": ["Number=Plur", "Person=1"],
    "P6": ["Number=Plur", "Person=2"],
    "P7": ["Number=Plur", "Person=3"],
    "P8": ["Number=Plur", "Person=2", "Polite=Form"],
}

# Transitional morphemes (X_Y) are keyed by their second part.
TRANSITION_MAP: Dict[str, List[str]] = {
    "KSE": ["VerbForm=Conv"],   # көсемше   / converb
    "ESM": ["VerbForm=Part"],   # есімше    / participle
    "ETU": ["VerbForm=Inf"],    # тұйық ет. / infinitive
    "ETB": ["Polarity=Neg"],    # болымсыз  / negated verb
}

# Deliberately unmapped KLC morphemes (derivational, or no gold counterpart).
UNMAPPED_MORPHEMES = frozenset({"S9", "LATT", "SML", "ABE", "EQU", "V4"})


def parse_analysis(analysis: str) -> Tuple[str, str, List[str]]:
    """Parse one disambiguated KazNLP analysis string into (lemma, klc_pos, morph_tags).

    Input looks like ``"алма_R_ZE сы_S3 н_C4"`` (segments separated by spaces;
    each segment is ``morpheme_TAG`` where ``TAG`` is everything after the
    first underscore). Returns the lemma (root morpheme of the first segment),
    the KLC POS tag of the first segment (e.g. ``"R_ZE"``) and the ordered
    list of the remaining grammatical-morpheme tags (e.g. ``["S3", "C4"]``).
    """
    segments = analysis.strip().split()
    if not segments:
        return "", "R_X", []
    first = segments[0]
    if "_" in first:
        lemma, pos = first.split("_", 1)
    else:
        lemma, pos = first, "R_X"
    morph_tags: List[str] = []
    for seg in segments[1:]:
        if "_" in seg:
            morph_tags.append(seg.split("_", 1)[1])
        else:
            morph_tags.append(seg)
    return lemma, pos, morph_tags


def map_analysis(form: str, analysis: str) -> Tuple[str, str, str, Dict[str, int]]:
    """Map a KazNLP analysis of ``form`` to a (lemma, upos, feats) triple.

    ``feats`` is a UD FEATS string ("Attr=Val|Attr=Val" or "_"), containing
    only values present in :data:`GOLD_GRAMMEMES` and sorted for determinism.
    The fourth return value is a small counter of dropped/unmapped events, so
    a caller can aggregate a coverage report over the whole corpus.
    """
    counters = {"dropped_non_gold": 0, "unmapped_morpheme": 0, "unknown_pos": 0}
    lemma, klc_pos, morph_tags = parse_analysis(analysis)
    if not lemma:
        lemma = form

    upos = POS_MAP.get(klc_pos)
    if upos is None:
        upos = "X"
        counters["unknown_pos"] += 1

    feats: List[str] = []
    saw_verbform = False
    saw_mood = False
    saw_case = False
    saw_number = False

    for tag in morph_tags:
        vals: Optional[List[str]] = None
        if tag in MORPH_MAP:
            vals = MORPH_MAP[tag]
        elif "_" in tag:                       # transitional morpheme X_Y
            second = tag.split("_", 1)[1]
            vals = TRANSITION_MAP.get(second)
        if vals is None:
            if tag not in UNMAPPED_MORPHEMES:
                counters["unmapped_morpheme"] += 1
            continue
        for v in vals:
            if v in GOLD_GRAMMEMES:
                feats.append(v)
                if v.startswith("VerbForm="):
                    saw_verbform = True
                if v.startswith("Mood="):
                    saw_mood = True
                if v.startswith("Case="):
                    saw_case = True
                if v.startswith("Number=") and not v.startswith("Number["):
                    saw_number = True
            else:
                counters["dropped_non_gold"] += 1

    # ---- unmarked-category defaults (HEURISTIC, UD-conventional) ----
    if upos in _NOMINAL_UPOS:
        if not saw_case:
            feats.append("Case=Nom")
        if not saw_number:
            feats.append("Number=Sing")
    if upos in _VERBAL_UPOS:
        has_finite_agreement = any(t in {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"} for t in morph_tags)
        if not saw_verbform and has_finite_agreement:
            feats.append("VerbForm=Fin")
            if not saw_mood:
                feats.append("Mood=Ind")

    # de-duplicate, keep only gold values, sort for determinism
    feats = sorted({f for f in feats if f in GOLD_GRAMMEMES})
    feats_str = "|".join(feats) if feats else "_"
    return lemma, upos, feats_str, counters
