from scoring import score_answer


def test_score_answer_numerical_match():
    assert score_answer("The result is 8", "8", "numerical")


def test_score_answer_numerical_mismatch():
    assert not score_answer("The result is 7", "8", "numerical")


def test_score_answer_short_answer_substring():
    assert score_answer("The output is AB", "AB", "short_answer")


def test_score_answer_short_answer_mismatch():
    assert not score_answer("The output is XY", "AB", "short_answer")


def test_score_answer_empty_ground_truth_is_never_correct():
    assert not score_answer("anything", "", "short_answer")
