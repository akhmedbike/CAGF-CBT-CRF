from cagf.kk_morphrules import guess_nominal_morphology
CASES = [('студенттер', 'Plur', None, 'студент'), ('оқушылар', 'Plur', None, 'оқушы'), ('мектепте', None, 'Loc', 'мектеп'), ('кітаптан', None, 'Abl', 'кітап'), ('үймен', None, 'Ins', 'үй'), ('қаланың', None, 'Gen', 'қала'), ('балаға', None, 'Dat', 'бала'), ('қыздарды', 'Plur', 'Acc', 'қыз'), ('елдің', None, 'Gen', 'ел'), ('баланы', None, 'Acc', 'бала'), ('үй', None, None, 'үй'), ('бала', None, None, 'бала')]

def test_nominal_morphology_regression_cases():
    for word, exp_num, exp_case, exp_lemma in CASES:
        g = guess_nominal_morphology(word)
        assert g.number == exp_num, f'{word}: expected number={exp_num}, got {g.number}'
        assert g.case == exp_case, f'{word}: expected case={exp_case}, got {g.case}'
        assert g.lemma_guess == exp_lemma, f'{word}: expected lemma={exp_lemma}, got {g.lemma_guess}'

def test_unmatched_token_is_low_confidence_not_guessed_nominative():
    g = guess_nominal_morphology('2026')
    assert g.confidence == 'low'
    assert g.case is None
    assert g.number is None
if __name__ == '__main__':
    test_nominal_morphology_regression_cases()
    test_unmatched_token_is_low_confidence_not_guessed_nominative()
    print('All kk_morphrules regression tests passed.')