from detector import check_groundedness


def test_check_groundedness_flags_unsupported_claim():
    contexts = ["Binary search runs in O(log n) time on a sorted array."]
    answer = "Binary search runs in O(log n) time. It was invented by aliens in 1998."
    result = check_groundedness(answer, contexts)
    assert result["flagged_count"] >= 1


def test_check_groundedness_grounded_answer_has_no_flags():
    contexts = ["Binary search runs in O(log n) time on a sorted array."]
    answer = "Binary search runs in O(log n) time on a sorted array."
    result = check_groundedness(answer, contexts)
    assert result["flagged_count"] == 0
    assert result["groundedness_rate"] == 1.0


def test_check_groundedness_empty_answer_is_fully_grounded_by_convention():
    result = check_groundedness("", ["some context"])
    assert result["claims"] == 0
    assert result["groundedness_rate"] == 1.0
