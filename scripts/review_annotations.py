from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path

def read_sentences(path: Path) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.strip() == '':
                if current:
                    blocks.append(current)
                    current = []
            else:
                current.append(line.rstrip('\n'))
    if current:
        blocks.append(current)
    return blocks

def parse_token_lines(block: list[str]) -> tuple[list[str], list[list[str]]]:
    comments = [l for l in block if l.startswith('#')]
    tokens = [l.split('\t') for l in block if not l.startswith('#')]
    return (comments, tokens)

def format_sentence_for_review(comments: list[str], tokens: list[list[str]]) -> str:
    lines = list(comments)
    lines.append('')
    for i, t in enumerate(tokens, start=1):
        form, lemma, upos, feats, misc = (t[1], t[2], t[3], t[5], t[9])
        marker = ''
        if 'PreAnnot=rule' in misc:
            conf = 'high' if 'Confidence=high' in misc else 'low'
            marker = f'  [draft, conf={conf}]'
        lines.append(f'  {i:3d}  {form:20s} lemma={lemma:15s} upos={upos:6s} feats={feats:20s}{marker}')
    return '\n'.join(lines)

def apply_corrections(tokens: list[list[str]], corrections: list[str], annotator: str) -> None:
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    correction_map = {}
    for line in corrections:
        parts = line.split()
        if len(parts) == 3:
            idx_str, lemma, feats = parts
            upos = None
        elif len(parts) == 4:
            idx_str, lemma, upos, feats = parts
        else:
            print(f"  (ignored malformed correction: {line!r} -- expected 'INDEX LEMMA FEATS' or 'INDEX LEMMA UPOS FEATS')")
            continue
        try:
            idx = int(idx_str)
        except ValueError:
            print(f'  (ignored malformed correction: {line!r} -- INDEX must be a number)')
            continue
        correction_map[idx] = (lemma, upos, feats)
    for i, t in enumerate(tokens, start=1):
        if i in correction_map:
            lemma, upos, feats = correction_map[i]
            t[2] = lemma
            if upos is not None:
                t[3] = upos
            t[5] = feats
            t[9] = f'Annotator={annotator}|Verified=yes|Corrected=yes|ReviewedAt={ts}'
        else:
            t[9] = f'Annotator={annotator}|Verified=yes|Corrected=no|ReviewedAt={ts}'

def load_progress(state_path: Path) -> int:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding='utf-8'))['next_index']
    return 0

def save_progress(state_path: Path, next_index: int) -> None:
    state_path.write_text(json.dumps({'next_index': next_index}), encoding='utf-8')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--annotator', required=True)
    ap.add_argument('--start-from', type=int, default=None, help='override the saved resume position')
    args = ap.parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    state_path = out_path.with_suffix('.progress.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sentences = read_sentences(in_path)
    start = args.start_from if args.start_from is not None else load_progress(state_path)
    print(f'Loaded {len(sentences)} sentences. Resuming from sentence #{start}.')
    print("Commands: <Enter>=accept all, s=skip, q=save&quit, or 'INDEX LEMMA FEATS' per correction.\n")
    out_fh = open(out_path, 'a', encoding='utf-8')
    i = start
    try:
        while i < len(sentences):
            comments, tokens = parse_token_lines(sentences[i])
            print(f'\n=== sentence {i + 1}/{len(sentences)} ===')
            print(format_sentence_for_review(comments, tokens))
            print('\nEnter corrections (blank line to submit), or s/q:')
            corrections = []
            action = None
            while True:
                line = input('> ').strip()
                if line == '' and (not corrections):
                    action = 'accept'
                    break
                if line == '':
                    action = 'submit'
                    break
                if line == 's':
                    action = 'skip'
                    break
                if line == 'q':
                    action = 'quit'
                    break
                corrections.append(line)
            if action == 'quit':
                save_progress(state_path, i)
                print(f'Progress saved at sentence #{i}. Resume with the same command later.')
                break
            if action == 'skip':
                i += 1
                save_progress(state_path, i)
                continue
            apply_corrections(tokens, corrections, args.annotator)
            for c in comments:
                out_fh.write(c + '\n')
            for t in tokens:
                out_fh.write('\t'.join(t) + '\n')
            out_fh.write('\n')
            out_fh.flush()
            i += 1
            save_progress(state_path, i)
    finally:
        out_fh.close()
    if i >= len(sentences):
        print(f'\nAll {len(sentences)} sentences reviewed. Gold output: {out_path}')
if __name__ == '__main__':
    main()