"""Model efficiency measurement: parameter count and inference latency.

Produces the numbers behind the paper's efficiency table (Reviewer 1 #4 /
Reviewer 3: "does not specify hardware, batch size, sequence length,
precision, implementation framework, or measurement protocol").

For each ablation configuration (and the baseline encoders), it measures:
  * trainable parameter count (torch.numel, the full model including embeddings)
  * inference latency for batch_size=1 (interactive, like the web UI) and
    batch_size=32 (batched processing), reported as median and IQR in ms/token,
    over a fixed measurement protocol (warmup + N timed iterations).

Hardware/framework/precision are recorded in the report header so the table
caption is self-describing.

Usage
-----
    PYTHONPATH=. python scripts/efficiency.py --out results/tables/efficiency.json
"""
from __future__ import annotations
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from cagf.data import build_vocabs, read_conllu
from cagf.device import pick_device
from cagf.model import AblationConfig, ModelHParams, CAGFCBTCRF


ABLATIONS = {
    'full_model': AblationConfig(),
    'wo_character_encoder': AblationConfig(use_char_cnn=False, use_char_bilstm=False),
    'wo_gated_fusion': AblationConfig(use_gated_fusion=False),
    'wo_crf': AblationConfig(use_crf=False),
    'transformer_only': AblationConfig(use_char_cnn=False, use_char_bilstm=False,
                                       use_word_bilstm=False, use_gated_fusion=False),
}


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_dummy_batch(vocab_sizes: dict, batch_size: int, seq_len: int, device: str):
    """Random ids of the right shape; lengths/mask consistent with seq_len."""
    word_ids = torch.randint(0, vocab_sizes['num_words'], (batch_size, seq_len), device=device)
    char_dim = 20  # matches MorphDataset's per-token char window
    char_ids = torch.randint(0, vocab_sizes['num_chars'], (batch_size, seq_len, char_dim), device=device)
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    return word_ids, char_ids, lengths, mask


def measure_latency(model: torch.nn.Module, batch: tuple, device: str,
                    warmup: int = 20, iters: int = 100) -> dict:
    model.eval()
    word_ids, char_ids, lengths, mask = batch
    n_tokens = int(mask.sum().item())
    if device == 'mps':
        sync = torch.mps.synchronize
    elif device == 'cuda':
        sync = torch.cuda.synchronize
    else:
        sync = lambda: None

    with torch.no_grad():
        for _ in range(warmup):
            model(word_ids, char_ids, lengths, mask, upos_ids=None)
        sync()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(word_ids, char_ids, lengths, mask, upos_ids=None)
            sync()
            times.append(time.perf_counter() - t0)
    per_token_ms = [(t / n_tokens) * 1000 for t in times]
    return {
        'n_tokens': n_tokens,
        'iters': iters,
        'latency_ms_per_token_median': statistics.median(per_token_ms),
        'latency_ms_per_token_iqr': _iqr(per_token_ms),
        'latency_ms_per_batch_median': statistics.median(times) * 1000,
    }


def _iqr(xs: list[float]) -> float:
    if len(xs) < 4:
        return 0.0
    s = sorted(xs)
    q1 = s[len(s) // 4]
    q3 = s[3 * len(s) // 4]
    return q3 - q1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gold-train', default='data/gold_merged/gold_train.conllu')
    ap.add_argument('--seq-len', type=int, default=32,
                    help='sequence length (mean KTB sentence = 9.77 tokens; 32 covers >99%%)')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--out', default='results/tables/efficiency.json')
    args = ap.parse_args()

    device = pick_device()
    train = read_conllu(args.gold_train)
    vocabs = build_vocabs(train, min_word_freq=1, max_word_vocab=50000)
    vs = {'num_chars': len(vocabs.char_vocab), 'num_words': len(vocabs.word_vocab),
          'num_upos': len(vocabs.upos_vocab), 'num_grammemes': len(vocabs.grammeme_vocab),
          'num_lemma_rules': len(vocabs.lemma_rule_vocab)}
    hp = ModelHParams()

    report = {
        'protocol': {
            'hardware': f'{platform.machine()} / {platform.platform()}',
            'device': device,
            'framework': f'torch {torch.__version__}',
            'precision': 'fp32',
            'sequence_length': args.seq_len,
            'warmup_iterations': args.warmup,
            'measured_iterations': args.iters,
            'param_count_method': 'torch.numel (trainable only)',
        },
        'configs': {},
    }

    print(f"Device: {device}  seq_len={args.seq_len}  warmup={args.warmup} iters={args.iters}")
    print(f"{'config':<24}{'params':>12}{'ms/tok b=1':>14}{'ms/tok b=32':>14}")
    print('-' * 64)
    for name, ablation in ABLATIONS.items():
        model = CAGFCBTCRF(num_chars=vs['num_chars'], num_words=vs['num_words'],
                           num_upos=vs['num_upos'], num_grammemes=vs['num_grammemes'],
                           num_lemma_rules=vs['num_lemma_rules'], hparams=hp, ablation=ablation,
                           char_pad_id=vocabs.char_vocab.stoi['<pad>'],
                           word_pad_id=vocabs.word_vocab.stoi['<pad>']).to(device)
        n_params = count_params(model)
        lat1 = measure_latency(model, make_dummy_batch(vs, 1, args.seq_len, device), device,
                               warmup=args.warmup, iters=args.iters)
        lat32 = measure_latency(model, make_dummy_batch(vs, 32, args.seq_len, device), device,
                                warmup=args.warmup, iters=args.iters)
        report['configs'][name] = {
            'trainable_params': n_params,
            'batch_1': lat1, 'batch_32': lat32,
        }
        print(f"{name:<24}{n_params:>12,}"
              f"{lat1['latency_ms_per_token_median']*1000:>11.2f} us"
              f"{lat32['latency_ms_per_token_median']*1000:>11.2f} us")
        del model

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
