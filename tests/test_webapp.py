import io
import os
import pytest
CHECKPOINT = 'models/interface_model.pt'
VOCABS = 'models/interface_vocabs.json'
pytestmark = pytest.mark.skipif(not (os.path.exists(CHECKPOINT) and os.path.exists(VOCABS)), reason='no trained checkpoint present -- run scripts/train_for_interface.py first')

@pytest.fixture(scope='module')
def client():
    import webapp.app as wa
    from cagf.inference import load_checkpoint
    wa.MODEL, wa.VOCABS, wa.META = load_checkpoint(CHECKPOINT, VOCABS)
    return wa.app.test_client()

def test_index_get(client):
    r = client.get('/')
    assert r.status_code == 200
    assert 'CAGF-CBT+CRF' in r.get_data(as_text=True)

def test_analyze_raw_text(client):
    r = client.post('/', data={'raw_text': 'Бүгін студенттер мектепке барды.'})
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'студенттер' in html
    assert 'Error' not in html

def test_analyze_txt_upload(client):
    data = {'file': (io.BytesIO('Мектепте оқушылар оқиды.'.encode('utf-8')), 'test.txt')}
    r = client.post('/', data=data, content_type='multipart/form-data')
    assert r.status_code == 200
    assert 'оқушылар' in r.get_data(as_text=True)

def test_analyze_conllu_upload(client):
    content = '# sent_id = 1\n1\tбала\t_\t_\t_\t_\t_\t_\t_\t_\n\n'
    data = {'file': (io.BytesIO(content.encode('utf-8')), 'test.conllu')}
    r = client.post('/', data=data, content_type='multipart/form-data')
    assert r.status_code == 200
    assert 'бала' in r.get_data(as_text=True)

def test_charts_and_export_present_after_analysis(client):
    long_text = 'Бүгін студенттер мектепке барды. Олар кітап оқыды және дәріс тыңдады.'
    r = client.post('/', data={'raw_text': long_text})
    html = r.get_data(as_text=True)
    assert 'metricsChart' not in html
    assert 'uposDistChart' in html
    assert 'confHistChart' in html
    assert 'exportConllu' in html
    assert 'not a CRF marginal probability' in html

def test_clear_button_present(client):
    r = client.get('/')
    html = r.get_data(as_text=True)
    assert 'clearForm' in html
    assert 'Clear' in html

def test_confidence_classification_thresholds():
    from webapp.app import _confidence_css_class
    assert _confidence_css_class(0.95) == 'conf-high'
    assert _confidence_css_class(0.8) == 'conf-high'
    assert _confidence_css_class(0.6) == 'conf-mid'
    assert _confidence_css_class(0.5) == 'conf-mid'
    assert _confidence_css_class(0.486) == 'conf-low'
    assert _confidence_css_class(0.1) == 'conf-low'

def test_charts_hidden_for_short_input(client):
    r = client.post('/', data={'raw_text': 'Балалар мектепке барды.'})
    html = r.get_data(as_text=True)
    assert 'uposDistChart' not in html
    assert 'Charts are hidden' in html

def test_charts_shown_for_long_input(client):
    long_text = 'Бүгін студенттер мектепке барды. Олар кітап оқыды және дәріс тыңдады.'
    r = client.post('/', data={'raw_text': long_text})
    html = r.get_data(as_text=True)
    assert 'uposDistChart' in html

def test_invalid_lemma_prediction_is_flagged_not_silently_masked(client):
    r = client.post('/', data={'raw_text': 'Кітап көпшілік оқырманға арналған.'})
    html = r.get_data(as_text=True)
    assert 'unparsed' in html

def test_interactive_pos_filter_and_sort_present(client):
    long_text = 'Бүгін студенттер мектепке барды. Олар кітап оқыды және дәріс тыңдады.'
    r = client.post('/', data={'raw_text': long_text})
    html = r.get_data(as_text=True)
    assert 'id="posFilter"' in html
    assert 'sortByConfidence' in html
    assert 'data-upos=' in html
    assert 'data-conf=' in html

def test_grammeme_distribution_chart_present_when_grammemes_found(client):
    long_text = 'Бүгін студенттер мектепке барды. Олар кітап оқыды және дәріс тыңдады.'
    r = client.post('/', data={'raw_text': long_text})
    html = r.get_data(as_text=True)
    assert 'grammemeDistChart' in html

def test_empty_input_does_not_crash(client):
    r = client.post('/', data={'raw_text': ''})
    assert r.status_code == 200
    assert 'Error' in r.get_data(as_text=True)

def test_training_status_reports_unavailable_when_no_results_file(client, tmp_path):
    import webapp.app as wa
    original_path = wa.ABLATION_RESULTS_PATH
    wa.ABLATION_RESULTS_PATH = tmp_path / 'does_not_exist.json'
    try:
        r = client.get('/training_status')
        data = r.get_json()
        assert data['available'] is False
        assert data['completed'] == 0
    finally:
        wa.ABLATION_RESULTS_PATH = original_path

def test_training_status_aggregates_real_results_file(client, tmp_path):
    import json as json_module
    import webapp.app as wa
    results_file = tmp_path / 'ablation_raw.json'
    results_file.write_text(json_module.dumps([
        {'config': 'full_model', 'seed': 13, 'lemma': {'f1': 0.42}, 'upos': {'f1': 0.75}, 'grammeme': {'f1': 0.30}},
        {'config': 'full_model', 'seed': 42, 'lemma': {'f1': 0.44}, 'upos': {'f1': 0.77}, 'grammeme': {'f1': 0.32}},
        {'config': 'wo_crf', 'seed': 13, 'lemma': {'f1': 0.38}, 'upos': {'f1': 0.70}, 'grammeme': {'f1': 0.28}},
    ]), encoding='utf-8')
    original_path = wa.ABLATION_RESULTS_PATH
    wa.ABLATION_RESULTS_PATH = results_file
    try:
        r = client.get('/training_status')
        data = r.get_json()
        assert data['available'] is True
        assert data['completed'] == 3
        assert data['expected'] == 25
        full_model_row = next(row for row in data['runs'] if row['config'] == 'full_model')
        assert full_model_row['seeds_done'] == 2
        assert abs(full_model_row['upos_f1'] - 0.76) < 1e-6
    finally:
        wa.ABLATION_RESULTS_PATH = original_path

def test_live_training_card_present_in_page(client):
    r = client.get('/')
    html = r.get_data(as_text=True)
    assert 'liveTrainingCard' in html
    assert 'pollTrainingStatus' in html