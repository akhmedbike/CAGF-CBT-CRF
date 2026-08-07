import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from cagf.baselines import BASELINE_CLASSES, BaselineHParams
from cagf.losses import masked_lemma_ce, masked_multilabel_bce


def _dummy_batch():
    B, T, W = 3, 6, 10
    word_ids = torch.randint(1, 50, (B, T))
    char_ids = torch.randint(1, 20, (B, T, W))
    upos_ids = torch.randint(0, 5, (B, T))
    lemma_ids = torch.randint(0, 12, (B, T))
    gram_targets = torch.randint(0, 2, (B, T, 8)).float()
    lengths = torch.tensor([6, 4, 5])
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    return word_ids, char_ids, upos_ids, lemma_ids, gram_targets, lengths, mask


def test_all_four_baselines_exist():
    assert set(BASELINE_CLASSES.keys()) == {"cnn", "cnn_bilstm", "cnn_bilstm_transformer", "subword_tagging"}


def test_all_baselines_forward_backward_with_full_multitask_loss():
    hp = BaselineHParams(hidden_dim=32, word_emb_dim=16, char_emb_dim=8, char_cnn_filters=8,
                          transformer_heads=2, transformer_layers=1)
    word_ids, char_ids, upos_ids, lemma_ids, gram_targets, lengths, mask = _dummy_batch()

    for name, cls in BASELINE_CLASSES.items():
        model = cls(num_chars=25, num_words=60, num_upos=5, num_grammemes=8, num_lemma_rules=12, hparams=hp)
        out = model(word_ids, char_ids, lengths, mask, upos_ids=upos_ids)
        assert "upos_nll" in out and "lemma_logits" in out and "grammeme_logits" in out
        assert torch.isfinite(out["upos_nll"])

        total = (out["upos_nll"]
                 + masked_lemma_ce(out["lemma_logits"], lemma_ids)
                 + masked_multilabel_bce(out["grammeme_logits"], gram_targets, mask))
        total.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad), \
            f"{name}: some parameters received no gradient"


def test_cnn_baseline_has_no_recurrent_or_attention_layers():
    """Regression test: the CNN baseline must be genuinely context-free
    (no BiLSTM, no Transformer) -- that is its defining property per the
    manuscript's own description of this configuration."""
    hp = BaselineHParams(hidden_dim=16, word_emb_dim=8, char_emb_dim=8)
    model = BASELINE_CLASSES["cnn"](num_chars=10, num_words=10, num_upos=5,
                                     num_grammemes=5, num_lemma_rules=5, hparams=hp)
    assert not hasattr(model, "bilstm")
    assert not hasattr(model, "transformer")


def test_subword_tagging_baseline_has_no_word_embeddings():
    """Regression test: SubwordTaggingBaseline must not use word-level
    embeddings -- its distinguishing feature is subword-only input."""
    hp = BaselineHParams(hidden_dim=16, word_emb_dim=8, char_emb_dim=8)
    model = BASELINE_CLASSES["subword_tagging"](num_chars=10, num_words=10, num_upos=5,
                                                  num_grammemes=5, num_lemma_rules=5, hparams=hp)
    assert not hasattr(model, "word_embed")
    assert hasattr(model, "bilstm")


if __name__ == "__main__":
    test_all_four_baselines_exist()
    test_all_baselines_forward_backward_with_full_multitask_loss()
    test_cnn_baseline_has_no_recurrent_or_attention_layers()
    test_subword_tagging_baseline_has_no_word_embeddings()
    print("All baseline tests passed.")
