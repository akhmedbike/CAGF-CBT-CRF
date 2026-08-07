import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prepare_raw_corpus import flag_reason, segment_sentences, tokenize

def test_front_matter_digit_case_still_flagged():
    sent = 'Прошу вас, верните ему Свободу » Луиза 1 Ұлы Отан соғысының жеңіспен аяқталғанына.'
    assert flag_reason(sent) == 'possible_front_matter_digit'

def test_watermark_case_now_flagged():
    sent = 'Қаһар t.me epub.kz БІРІНШІ БӨЛІМ І Ай сәулесі түсіп тұр.'
    assert flag_reason(sent) == 'possible_url_watermark'

def test_ordinary_sentence_not_flagged():
    sent = 'Бүгін студенттер мектепке барды.'
    assert flag_reason(sent) is None

def test_tokenize_and_segment_still_work_after_edit():
    sentences = segment_sentences('Бүгін жаңбыр жауды. Ол үйде отырды.')
    assert len(sentences) == 2
    tokens = tokenize(sentences[0])
    assert tokens[0] == 'Бүгін'
if __name__ == '__main__':
    test_front_matter_digit_case_still_flagged()
    test_watermark_case_now_flagged()
    test_ordinary_sentence_not_flagged()
    test_tokenize_and_segment_still_work_after_edit()
    print('All prepare_raw_corpus regression tests passed.')