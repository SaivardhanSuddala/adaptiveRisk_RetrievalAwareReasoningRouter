import random

from generation import STRATEGY_FUNCTIONS
from retrieval import retrieval_quality, retrieve
from router import ROUTING_WEIGHTS, estimate_hallucination_risk


STRATEGIES = tuple(STRATEGY_FUNCTIONS.keys())


def route_random(rng: random.Random) -> str:
    return rng.choice(STRATEGIES)


def route_risk_only(risk: float) -> str:
    if risk < 0.3:
        return "direct"
    if risk < 0.5:
        return "cot"
    if risk < 0.7:
        return "react"
    if risk < 0.85:
        return "reflection"
    return "self_consistency"


def route_retrieval_only(quality_score: float) -> str:
    gap = 1 - quality_score

    if gap < 0.3:
        return "direct"
    if gap < 0.5:
        return "cot"
    if gap < 0.7:
        return "react"
    if gap < 0.85:
        return "reflection"
    return "self_consistency"


def route_full(risk: float, quality_score: float) -> str:
    w1, w2 = ROUTING_WEIGHTS
    combined = w1 * risk + w2 * (1 - quality_score)

    if combined < 0.3:
        return "direct"
    if combined < 0.5:
        return "cot"
    if combined < 0.7:
        return "react"
    if combined < 0.85:
        return "reflection"
    return "self_consistency"


def run_ablation(rows: list[dict], seed: int = 13) -> list[dict]:
    rng = random.Random(seed)
    output = []

    for row in rows:
        query = row["question"]
        results = retrieve(query)
        quality = retrieval_quality(results, query)
        risk = estimate_hallucination_risk(query)

        output.append({
            "question_id": row["question_id"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "hallucination_risk": risk,
            "retrieval_quality": quality["score"],
            "router_only_random": route_random(rng),
            "risk_predictor_only": route_risk_only(risk),
            "retrieval_estimator_only": route_retrieval_only(quality["score"]),
            "full_arrr": route_full(risk, quality["score"]),
        })

    return output
