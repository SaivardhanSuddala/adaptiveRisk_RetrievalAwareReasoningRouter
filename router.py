from dataclasses import dataclass, asdict
import re

from configs import TOP_K
from generation import generate_answer
from retrieval import retrieval_quality, retrieve


@dataclass
class RouteDecision:
    strategy: str
    hallucination_risk: float
    retrieval_quality: float
    combined_score: float
    use_retrieval: bool
    retrieve_again: bool
    reason: str


HIGH_RISK_TERMS = {
    "prove",
    "derive",
    "debug",
    "optimize",
    "complexity",
    "concurrency",
    "deadlock",
    "probability",
    "gradient",
    "determinant",
    "least-squares",
    "recursion",
    "algorithm",
    "why",
    "explain",
}

CONSTRAINT_TERMS = {
    "must",
    "should",
    "cannot",
    "can't",
    "at least",
    "at most",
    "exactly",
    "only",
    "unless",
    "except",
}

MULTISTEP_TERMS = {
    "then",
    "after",
    "next",
    "finally",
    "and then",
    "step",
}

TECHNICAL_TERMS = {
    "algorithm",
    "complexity",
    "recursion",
    "concurrency",
    "deadlock",
    "scheduler",
    "gradient",
    "matrix",
    "determinant",
    "probability",
    "kernel",
    "syscall",
    "thread",
    "mutex",
    "pointer",
    "compiler",
    "runtime",
    "database",
    "regression",
    "classifier",
    "hashmap",
    "heap",
    "stack",
}

ROUTING_WEIGHTS = (0.5, 0.5)


def _count_terms(lowered: str, terms: set[str]) -> int:
    return sum(1 for term in terms if term in lowered)


def _entity_count(query: str) -> int:
    return len(re.findall(r"\b[A-Z][a-zA-Z0-9_]*\b", query))


def estimate_hallucination_risk(query: str) -> float:
    lowered = query.lower()
    score = 0.15

    if len(query) > 180:
        score += 0.15
    elif len(query) > 100:
        score += 0.08

    if any(term in lowered for term in HIGH_RISK_TERMS):
        score += 0.2

    if re.search(r"\b(code|python|c\+\+|java|sql|matrix|equation|runtime|memory)\b", lowered):
        score += 0.12

    constraint_count = _count_terms(lowered, CONSTRAINT_TERMS)
    score += min(constraint_count * 0.05, 0.15)

    multistep_signal = (
        _count_terms(lowered, MULTISTEP_TERMS)
        + lowered.count(";")
        + max(query.count(".") - 1, 0)
    )
    score += min(multistep_signal * 0.04, 0.15)

    score += min(_entity_count(query) * 0.02, 0.1)

    if _count_terms(lowered, TECHNICAL_TERMS) > 0:
        score += 0.08

    if query.count("?") > 1:
        score += 0.05

    if re.search(r"\d", query):
        score += 0.05

    return round(min(score, 1.0), 4)


def choose_strategy(query: str, quality_score: float) -> RouteDecision:
    risk = estimate_hallucination_risk(query)
    w1, w2 = ROUTING_WEIGHTS
    combined = round(w1 * risk + w2 * (1 - quality_score), 4)

    retrieve_again = risk <= 0.35 and quality_score < 0.3

    if combined < 0.3:
        strategy, reason = "direct", "low combined risk; cheap strategy is sufficient"
    elif combined < 0.5:
        strategy, reason = "cot", "moderate risk warrants an explicit reasoning trace"
    elif combined < 0.7:
        strategy, reason = "react", "high risk or weak retrieval; ground the answer with tool lookups"
    elif combined < 0.85:
        strategy, reason = "reflection", "high risk; self-critique pass before committing"
    else:
        strategy, reason = "self_consistency", "highest risk; vote across independent reasoning paths"

    use_retrieval = quality_score >= 0.35 or strategy == "react"

    return RouteDecision(
        strategy=strategy,
        hallucination_risk=risk,
        retrieval_quality=quality_score,
        combined_score=combined,
        use_retrieval=use_retrieval,
        retrieve_again=retrieve_again,
        reason=reason,
    )


def answer_query(query: str, top_k: int = TOP_K, model: str | None = None) -> dict:
    retrieval_results = retrieve(query, top_k=top_k)
    quality = retrieval_quality(retrieval_results, query)
    decision = choose_strategy(query, quality["score"])

    if decision.retrieve_again:
        retrieval_results = retrieve(query, top_k=top_k * 2)
        quality = retrieval_quality(retrieval_results, query)

    contexts = retrieval_results["documents"][0] if decision.use_retrieval else []
    generation = generate_answer(query, contexts, decision.strategy, model=model)

    return {
        "query": query,
        "decision": asdict(decision),
        "retrieval": quality,
        "contexts": contexts,
        "generation": generation,
    }


def answer_query_with_strategy(
    query: str,
    strategy: str,
    top_k: int = TOP_K,
    model: str | None = None,
) -> dict:

    retrieval_results = retrieve(query, top_k=top_k)
    quality = retrieval_quality(retrieval_results, query)
    use_retrieval = quality["score"] >= 0.35
    contexts = retrieval_results["documents"][0] if use_retrieval else []
    generation = generate_answer(query, contexts, strategy, model=model)

    return {
        "query": query,
        "decision": {
            "strategy": strategy,
            "hallucination_risk": estimate_hallucination_risk(query),
            "retrieval_quality": quality["score"],
            "use_retrieval": use_retrieval,
            "forced": True,
        },
        "retrieval": quality,
        "contexts": contexts,
        "generation": generation,
    }
