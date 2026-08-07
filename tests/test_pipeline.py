import torch
from cagf.data import read_conllu, build_vocabs, form_to_edit_script, apply_edit_script
from cagf.model import AblationConfig, CAGFCBTCRF, ModelHParams
TOY_TRAIN = 'data/toy_train.conllu'

def test_conllu_roundtrip_lemma_rules():
    sents = read_conllu(TOY_TRAIN)
    for sent in sents:
        for tok in sent.tokens:
            script = form_to_edit_script(tok.form, tok.lemma)
            reconstructed = apply_edit_script(tok.form, script)
            assert reconstructed == tok.lemma.lower()

def test_all_ablation_configs_train_step():
    sents = read_conllu(TOY_TRAIN)
    vocabs = build_vocabs(sents, min_word_freq=1)
    hp = ModelHParams(hidden_dim=16, word_emb_dim=8, char_emb_dim=8, char_cnn_filters=8, transformer_heads=2, transformer_layers=1, char_bilstm_hidden=8)
    configs = [AblationConfig(), AblationConfig(use_char_cnn=False, use_char_bilstm=False), AblationConfig(use_gated_fusion=False), AblationConfig(use_crf=False), AblationConfig(use_char_cnn=False, use_char_bilstm=False, use_word_bilstm=False, use_gated_fusion=False)]
    expected_names = {'full_model', 'wo_character_encoder', 'wo_gated_fusion', 'wo_crf', 'transformer_only'}
    seen_names = set()
    B, T, W = (2, 3, 8)
    word_ids = torch.randint(1, min(len(vocabs.word_vocab), 20), (B, T))
    char_ids = torch.randint(1, min(len(vocabs.char_vocab), 15), (B, T, W))
    upos_ids = torch.randint(0, len(vocabs.upos_vocab), (B, T))
    lengths = torch.tensor([3, 2])
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    for ab in configs:
        seen_names.add(ab.name())
        model = CAGFCBTCRF(num_chars=len(vocabs.char_vocab), num_words=len(vocabs.word_vocab), num_upos=len(vocabs.upos_vocab), num_grammemes=len(vocabs.grammeme_vocab), num_lemma_rules=len(vocabs.lemma_rule_vocab), hparams=hp, ablation=ab)
        out = model(word_ids, char_ids, lengths, mask, upos_ids=upos_ids)
        assert 'upos_nll' in out
        assert torch.isfinite(out['upos_nll'])
        out['upos_nll'].backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any((g is not None for g in grads))
    assert seen_names == expected_names, f'missing configs: {expected_names - seen_names}'

def test_transformer_only_config_has_no_bilstm_or_char_params():
    ab = AblationConfig(use_char_cnn=False, use_char_bilstm=False, use_word_bilstm=False, use_gated_fusion=False)
    assert ab.name() == 'transformer_only'
    hp = ModelHParams(hidden_dim=16, word_emb_dim=8, char_emb_dim=8, transformer_heads=2, transformer_layers=1)
    model = CAGFCBTCRF(num_chars=10, num_words=10, num_upos=5, num_grammemes=5, num_lemma_rules=5, hparams=hp, ablation=ab)
    assert not hasattr(model, 'char_cnn')
    assert not hasattr(model, 'char_bilstm')
    assert not hasattr(model, 'bilstm')
    assert not hasattr(model, 'fusion')
    assert hasattr(model, 'transformer')
    assert hasattr(model, 'crf')
if __name__ == '__main__':
    test_conllu_roundtrip_lemma_rules()
    test_all_ablation_configs_train_step()
    test_transformer_only_config_has_no_bilstm_or_char_params()
    print('All pipeline tests passed.')