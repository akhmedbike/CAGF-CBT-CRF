from __future__ import annotations
import argparse
import re
from pathlib import Path
NON_BREAKING_ABBREVIATIONS = {'т.б', 'т.с.с', 'ж.б', 'б.з.д', 'б.з', 'және т.б', 'мыс', 'қар', 'қ', 'б', 'г', 'ж'}
SENTENCE_END_RE = re.compile('([.!?…]+)(\\s+|$)')
TOKEN_RE = re.compile('«|»|\\"|\'|[\\(\\)\\[\\]\\{\\}]|[.,!?;:…]+|[-–—]|[A-Za-zА-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ0-9]+(?:[\'’ʼ][A-Za-zА-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]+)*', re.UNICODE)

def segment_sentences(text: str) -> list[str]:
    text = re.sub('\\s+', ' ', text).strip()
    if not text:
        return []
    sentences = []
    last_end = 0
    for match in SENTENCE_END_RE.finditer(text):
        end = match.end()
        candidate = text[last_end:end].strip()
        if not candidate:
            continue
        preceding = text[last_end:match.start()].strip().split()
        last_word = preceding[-1] if preceding else ''
        if last_word.rstrip('.') in NON_BREAKING_ABBREVIATIONS:
            continue
        sentences.append(candidate)
        last_end = end
    tail = text[last_end:].strip()
    if tail:
        sentences.append(tail)
    return sentences

def tokenize(sentence: str) -> list[str]:
    return TOKEN_RE.findall(sentence)
KAZAKH_SPECIFIC_LETTERS = set('әіңғүұқөһӘІҢҒҮҰҚӨҺ')
FRONT_MATTER_FLAG_RE = re.compile('^[^' + ''.join(KAZAKH_SPECIFIC_LETTERS) + ']{0,80}\\d+\\s')
WATERMARK_FLAG_RE = re.compile('\\b[a-zA-Z0-9]{1,10}\\.[a-zA-Z]{2,6}\\b')
FRONT_MATTER_CHECK_WINDOW = 3

def flag_reason(sentence: str) -> str | None:
    if FRONT_MATTER_FLAG_RE.match(sentence):
        return 'possible_front_matter_digit'
    if WATERMARK_FLAG_RE.search(sentence):
        return 'possible_url_watermark'
    return None

def process_file(path: Path) -> list[tuple[list[str], str | None]]:
    text = path.read_text(encoding='utf-8')
    sentences = segment_sentences(text)
    results = []
    for i, sent in enumerate(sentences):
        tokens = tokenize(sent)
        if not tokens:
            continue
        reason = flag_reason(sent) if i < FRONT_MATTER_CHECK_WINDOW else None
        results.append((tokens, reason))
    return results

def write_conllu_skeleton(all_sentences: list[tuple[str, list[str], str | None]], out_path: Path) -> None:
    with open(out_path, 'w', encoding='utf-8') as fh:
        for sent_id, (source_file, tokens, reason) in enumerate(all_sentences, start=1):
            fh.write(f'# sent_id = {sent_id}\n')
            fh.write(f'# source = {source_file}\n')
            if reason:
                fh.write(f'# review_flag = {reason} -- verify manually\n')
            text_line = ' '.join(tokens)
            fh.write(f'# text = {text_line}\n')
            for tid, tok in enumerate(tokens, start=1):
                fh.write(f'{tid}\t{tok}\t_\t_\t_\t_\t_\t_\t_\t_\n')
            fh.write('\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--min-tokens', type=int, default=2, help='drop sentences shorter than this (likely segmentation noise)')
    args = ap.parse_args()
    input_dir = Path(args.input_dir)
    txt_files = sorted(input_dir.rglob('*.txt'))
    if not txt_files:
        raise SystemExit(f'No .txt files found under {input_dir}')
    all_sentences: list[tuple[str, list[str], str | None]] = []
    flagged_list: list[tuple[str, str]] = []
    for path in txt_files:
        results = process_file(path)
        results = [(t, r) for t, r in results if len(t) >= args.min_tokens]
        for tokens, reason in results:
            all_sentences.append((path.name, tokens, reason))
            if reason:
                flagged_list.append((path.name, reason))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_conllu_skeleton(all_sentences, out_path)
    total_sentences = len(all_sentences)
    total_tokens = sum((len(t) for _, t, _ in all_sentences))
    print(f'Processed {len(txt_files)} files')
    print(f'Total sentences: {total_sentences}')
    print(f'Total tokens:    {total_tokens}')
    print(f'Flagged for manual review: {len(flagged_list)}')
    for name, reason in flagged_list:
        print(f'  - {name} ({reason})')
    print(f'Output written to: {out_path}')
    print("\nNOTE: LEMMA / UPOS / FEATS columns are all '_' (unannotated).")
    print('This file is a SKELETON for annotation, not a labeled dataset.')
    print('All source files consist of long, unbroken paragraph-length lines')
    print('with no separate short lines for titles, chapter numbers, or')
    print('epigraphs -- so front matter cannot be reliably auto-stripped by')
    print('line structure. Sentences flagged above must be checked by hand')
    print('before annotation begins; the rest of the sentence segmentation')
    print('should also be spot-checked by a native speaker.')
if __name__ == '__main__':
    main()