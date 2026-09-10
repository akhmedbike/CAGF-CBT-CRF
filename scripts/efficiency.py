"""Model efficiency measurement for the revision's R3.11 answer.

Produces the numbers behind the paper's efficiency table: parameter count,
on-disk checkpoint size, allocator memory during train steps, inference
latency (batch 1 and batch 32) on the accelerator AND on CPU, empirical
training time from the actual CV artifacts, and the CRF's share of
parameters and decoding latency (R2 #6).

Measured systems: the five CAGF ablations plus the two pretrained-encoder
baselines actually used in the paper's protocol — KazRoBERTa and XLM-R
(pinned revisions), both in the same finetune-everything configuration as
the CV runs.

Hardware/framework/precision/power-source are recorded in the report header
so the table caption is self-describing. Weak/constrained hardware was not
available; this is stated in the report rather than papered over.

Usage
-----
    PYTHONPATH=. python scripts/efficiency.py --out results/tables/efficiency.json
"""
from __future__ import annotations
import argparse
import json
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from cagf.data import build_vocabs, read_conllu
from cagf.device import pick_device
from cagf.losses import masked_multilabel_bce, masked_lemma_ce
from cagf.model import AblationConfig, ModelHParams, CAGFCBTCRF
from cagf.hf_model import HFMorphModel

KAZR = 'kz-transformers/kaz-roberta-conversational'
KAZR_REV = '43077c2fd0a163487ed468b5ec3b8750686a5888'
XLMR = 'FacebookAI/xlm-roberta-base'
XLMR_REV = 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'


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


def power_source() -> str:
    try:
        out = subprocess.run(['pmset', '-g', 'batt'], capture_output=True,
                             text=True, timeout=5).stdout
        return 'AC' if 'AC Power' in out else 'battery'
    except Exception:
        return 'unknown'


def real_string_batch(sentences, batch_size: int, seq_len: int):
    """First `batch_size` gold sentences as (list[list[str]], bool mask).

    The HF models consume raw word strings; real Kazakh words keep the
    tokenizer load representative instead of random ids."""
    batch, mask = [], []
    for s in sentences[:batch_size]:
        words = [t.form for t in s.tokens][:seq_len]
        batch.append(words)
        mask.append([True] * len(words))
    max_len = max(len(w) for w in batch)
    for w, m in zip(batch, mask):
        pad = max_len - len(w)
        w += ['<pad>'] * pad
        m += [False] * pad
    return batch, torch.tensor(mask)


def measure_latency_hf(model, batch_words, mask, device, warmup=20, iters=100) -> dict:
    model.eval()
    mask = mask.to(next(model.parameters()).device)
    n_tokens = int(mask.sum().item())
    sync = torch.mps.synchronize if device == 'mps' else (torch.cuda.synchronize if device == 'cuda' else (lambda: None))
    with torch.no_grad():
        for _ in range(warmup):
            model(batch_words, mask, upos_ids=None)
        sync()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(batch_words, mask, upos_ids=None)
            sync()
            times.append(time.perf_counter() - t0)
    per_token_ms = [(t / n_tokens) * 1000 for t in times]
    return {
        'n_tokens': n_tokens, 'iters': iters,
        'latency_ms_per_token_median': statistics.median(per_token_ms),
        'latency_ms_per_token_iqr': _iqr(per_token_ms),
        'latency_ms_per_batch_median': statistics.median(times) * 1000,
    }


def state_dict_mb(model) -> float:
    """On-disk checkpoint size (fp32 state_dict, same format the CV runs save)."""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model.state_dict(), f.name)
        mb = Path(f.name).stat().st_size / 1e6
    Path(f.name).unlink(missing_ok=True)
    return mb


def train_step_memory(fwd_fn, device, steps: int = 3) -> dict:
    """Allocator memory (GB) around forward+backward train steps at the CV
    protocol batch size 16. `fwd_fn` zeroes grads, runs one forward and
    returns the summed task loss; the caller captures the MPS allocator size
    after the forward and after the backward of each step. On MPS this is the
    sustained training footprint rather than a strict instantaneous peak."""
    after_fwd = after_bwd = None
    for _ in range(steps):
        loss = fwd_fn()
        if device == 'mps':
            torch.mps.synchronize()
            after_fwd = torch.mps.current_allocated_memory() / 1e9
        loss.backward()
        if device == 'mps':
            torch.mps.synchronize()
            after_bwd = torch.mps.current_allocated_memory() / 1e9
    return {'allocator_gb_after_forward': after_fwd,
            'allocator_gb_after_backward': after_bwd,
            'batch_size': 16, 'steps': steps}


def training_time_from_artifacts() -> dict:
    """Empirical wall-clock per fold (mean over folds) from the actual CV
    runs behind the paper's tables — no synthetic re-benchmarking."""
    import statistics as st
    out = {}
    for label, path in (
        ('cagf_gold', 'results_cv_silver/stratified/ktb_only/per_fold.json'),
        ('cagf_silver', 'results_cv_silver/stratified/silver_unfiltered_pretrain_ktb_finetune/per_fold.json'),
        ('kazroberta_gold', 'results_cv_kazroberta/gold_only/per_fold.json'),
        ('kazroberta_silver', 'results_cv_kazroberta/silver_finetune/per_fold.json'),
        ('xlmr_gold', 'results_cv_xlmr/stratified_gold/gold_only/per_fold.json'),
        ('xlmr_silver', 'results_cv_xlmr/stratified_silver/silver_finetune/per_fold.json'),
    ):
        try:
            entries = json.loads(Path(path).read_text(encoding='utf-8'))
            out[label] = {'n_folds': len(entries),
                          'mean_sec_per_fold': st.mean(e['elapsed_sec'] for e in entries)}
        except FileNotFoundError:
            out[label] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gold-train', default='data/gold_merged/gold_train.conllu')
    ap.add_argument('--gold-test', default='data/gold_merged/gold_test.conllu',
                    help='real sentences for the HF-tokenizer latency batches')
    ap.add_argument('--seq-len', type=int, default=32,
                    help='sequence length (mean KTB sentence = 9.77 tokens; 32 covers >99%%)')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--iters', type=int, default=100)
    ap.add_argument('--cpu-iters', type=int, default=20,
                    help='CPU latency iterations (slower; same warmup protocol)')
    ap.add_argument('--out', default='results/tables/efficiency.json')
    args = ap.parse_args()

    device = pick_device()
    train = read_conllu(args.gold_train)
    test = read_conllu(args.gold_test)
    vocabs = build_vocabs(train, min_word_freq=1, max_word_vocab=50000)
    vs = {'num_chars': len(vocabs.char_vocab), 'num_words': len(vocabs.word_vocab),
          'num_upos': len(vocabs.upos_vocab), 'num_grammemes': len(vocabs.grammeme_vocab),
          'num_lemma_rules': len(vocabs.lemma_rule_vocab)}
    hp = ModelHParams()

    report = {
        'protocol': {
            'hardware': f'{platform.machine()} / {platform.platform()}',
            'device': device,
            'power_source_at_measurement': power_source(),
            'framework': f'torch {torch.__version__}',
            'precision': 'fp32',
            'sequence_length': args.seq_len,
            'warmup_iterations': args.warmup,
            'measured_iterations': {'accelerator': args.iters, 'cpu': args.cpu_iters},
            'param_count_method': 'torch.numel (trainable only)',
            'checkpoint_size_method': 'torch.save(state_dict) to temp file',
            'memory_method': 'MPS allocator size after forward / after backward '
                             'of train steps at the CV protocol batch size 16 '
                             '(sustained footprint, not strict peak)',
            'constrained_hardware': 'not available; stated honestly per R3.11',
        },
        'configs': {},
        'pretrained_encoders': {},
        'crf_share': {},
        'training_time_from_artifacts': training_time_from_artifacts(),
    }

    print(f"Device: {device} (power: {report['protocol']['power_source_at_measurement']})  "
          f"seq_len={args.seq_len}  warmup={args.warmup} iters={args.iters}")
    print(f"{'config':<24}{'params':>12}{'MB disk':>10}{'ms/tok b=1':>12}{'ms/tok b=32':>13}")
    print('-' * 71)
    for name, ablation in ABLATIONS.items():
        model = CAGFCBTCRF(num_chars=vs['num_chars'], num_words=vs['num_words'],
                           num_upos=vs['num_upos'], num_grammemes=vs['num_grammemes'],
                           num_lemma_rules=vs['num_lemma_rules'], hparams=hp, ablation=ablation,
                           char_pad_id=vocabs.char_vocab.stoi['<pad>'],
                           word_pad_id=vocabs.word_vocab.stoi['<pad>']).to(device)
        n_params = count_params(model)
        mb = state_dict_mb(model)
        lat1 = measure_latency(model, make_dummy_batch(vs, 1, args.seq_len, device), device,
                               warmup=args.warmup, iters=args.iters)
        lat32 = measure_latency(model, make_dummy_batch(vs, 32, args.seq_len, device), device,
                                warmup=args.warmup, iters=args.iters)
        entry = {'trainable_params': n_params, 'checkpoint_mb': mb,
                 'batch_1': lat1, 'batch_32': lat32}
        if name == 'full_model':
            # training-step allocator footprint + CPU latency (R3.11)
            b = make_dummy_batch(vs, 16, args.seq_len, device)
            word_ids, char_ids, lengths, mask = b
            upos_ids = torch.randint(0, vs['num_upos'], mask.shape, device=device)
            lemma_ids = torch.randint(0, vs['num_lemma_rules'], mask.shape, device=device)
            gram_t = (torch.rand(*mask.shape, vs['num_grammemes'], device=device) < 0.05).float()
            def fwd():
                model.zero_grad(set_to_none=True)
                out = model(word_ids, char_ids, lengths, mask, upos_ids=upos_ids)
                return (masked_lemma_ce(out['lemma_logits'], lemma_ids)
                        + masked_multilabel_bce(out['grammeme_logits'], gram_t, mask)
                        + out['upos_nll'])
            entry['train_step_memory'] = train_step_memory(fwd, device)
            cpu_model = CAGFCBTCRF(num_chars=vs['num_chars'], num_words=vs['num_words'],
                                   num_upos=vs['num_upos'], num_grammemes=vs['num_grammemes'],
                                   num_lemma_rules=vs['num_lemma_rules'], hparams=hp,
                                   ablation=AblationConfig(),
                                   char_pad_id=vocabs.char_vocab.stoi['<pad>'],
                                   word_pad_id=vocabs.word_vocab.stoi['<pad>'])
            entry['cpu_batch_1'] = measure_latency(cpu_model, make_dummy_batch(vs, 1, args.seq_len, 'cpu'),
                                                   'cpu', warmup=5, iters=args.cpu_iters)
            del cpu_model
        report['configs'][name] = entry
        print(f"{name:<24}{n_params:>12,}{mb:>9.1f}"
              f"{lat1['latency_ms_per_token_median']:>11.3f}"
              f"{lat32['latency_ms_per_token_median']:>12.3f}")
        del model

    # ---- CRF share (R2 #6): params + batch-1 decoding latency of the CRF ----
    full, wocrf = report['configs']['full_model'], report['configs']['wo_crf']
    report['crf_share'] = {
        'param_delta': full['trainable_params'] - wocrf['trainable_params'],
        'param_delta_pct': 100 * (full['trainable_params'] - wocrf['trainable_params'])
                           / wocrf['trainable_params'],
        'batch1_ms_per_token_delta': full['batch_1']['latency_ms_per_token_median']
                                     - wocrf['batch_1']['latency_ms_per_token_median'],
        'batch1_ms_per_token_delta_pct': 100 * (
            full['batch_1']['latency_ms_per_token_median']
            / wocrf['batch_1']['latency_ms_per_token_median'] - 1),
    }

    # ---- pretrained-encoder baselines (finetune-everything, as in the CV runs) ----
    print('-' * 71)
    for name, mn, rev in (('kazroberta', KAZR, KAZR_REV), ('xlm_roberta', XLMR, XLMR_REV)):
        model = HFMorphModel(num_upos=vs['num_upos'], num_grammemes=vs['num_grammemes'],
                             num_lemma_rules=vs['num_lemma_rules'], model_name=mn,
                             revision=rev, hidden_dim=256, use_crf=True,
                             freeze_encoder=False).to(device)
        total = count_params(model)
        enc = count_params(model.encoder)
        mb = state_dict_mb(model)
        lat1 = measure_latency_hf(model, *real_string_batch(test, 1, args.seq_len), device,
                                  warmup=args.warmup, iters=args.iters)
        lat32 = measure_latency_hf(model, *real_string_batch(test, 32, args.seq_len), device,
                                   warmup=args.warmup, iters=args.iters)
        words16, mask16 = real_string_batch(test, 16, args.seq_len)
        mask16 = mask16.to(device)
        upos16 = torch.randint(0, vs['num_upos'], mask16.shape, device=device)
        lemma16 = torch.randint(0, vs['num_lemma_rules'], mask16.shape, device=device)
        gram16 = (torch.rand(*mask16.shape, vs['num_grammemes'], device=device) < 0.05).float()
        def fwd_hf():
            model.zero_grad(set_to_none=True)
            out = model(words16, mask16, upos_ids=upos16)
            return (masked_lemma_ce(out['lemma_logits'], lemma16)
                    + masked_multilabel_bce(out['grammeme_logits'], gram16, mask16)
                    + out['upos_nll'])
        mem = train_step_memory(fwd_hf, device)
        cpu_model = HFMorphModel(num_upos=vs['num_upos'], num_grammemes=vs['num_grammemes'],
                                 num_lemma_rules=vs['num_lemma_rules'], model_name=mn,
                                 revision=rev, use_crf=True, freeze_encoder=False)
        cpu_lat1 = measure_latency_hf(cpu_model, *real_string_batch(test, 1, args.seq_len),
                                      'cpu', warmup=3, iters=args.cpu_iters)
        del cpu_model
        report['pretrained_encoders'][name] = {
            'model_name': mn, 'revision': rev,
            'trainable_params': total, 'encoder_params': enc,
            'task_heads_params': total - enc, 'checkpoint_mb': mb,
            'batch_1': lat1, 'batch_32': lat32, 'cpu_batch_1': cpu_lat1,
            'train_step_memory': mem,
        }
        print(f"{name:<24}{total:>12,}{mb:>9.1f}"
              f"{lat1['latency_ms_per_token_median']:>11.3f}"
              f"{lat32['latency_ms_per_token_median']:>12.3f}")
        del model

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nWrote {out}")
    print(f"CRF share: {report['crf_share']['param_delta']:,} params "
          f"(+{report['crf_share']['param_delta_pct']:.1f}%), "
          f"batch-1 latency {report['crf_share']['batch1_ms_per_token_delta_pct']:+.1f}%")


if __name__ == '__main__':
    main()
