from router import choose_strategy, estimate_hallucination_risk


def test_estimate_hallucination_risk_bounds():
    risk = estimate_hallucination_risk("What is 2 + 2?")
    assert 0.0 <= risk <= 1.0


def test_estimate_hallucination_risk_higher_for_complex_query():
    simple = estimate_hallucination_risk("What is a variable?")
    complex_query = estimate_hallucination_risk(
        "Prove the time complexity of quicksort and explain why the recursion "
        "causes a deadlock in concurrent execution; also derive the gradient."
    )
    assert complex_query > simple


def test_choose_strategy_low_risk_high_quality_routes_direct():
    decision = choose_strategy("What is a variable?", quality_score=0.9)
    assert decision.strategy == "direct"


def test_choose_strategy_high_risk_low_quality_routes_expensive():
    decision = choose_strategy(
        "Prove the amortized complexity and derive the recursion for concurrent deadlock probability",
        quality_score=0.05,
    )
    assert decision.strategy in {"reflection", "self_consistency", "react"}


def test_choose_strategy_combined_score_bounded():
    decision = choose_strategy("Explain recursion", quality_score=0.5)
    assert 0.0 <= decision.combined_score <= 1.0


def test_choose_strategy_flags_retrieve_again_when_signals_are_weak():
    decision = choose_strategy("What color is the sky?", quality_score=0.1)
    assert decision.retrieve_again is True
