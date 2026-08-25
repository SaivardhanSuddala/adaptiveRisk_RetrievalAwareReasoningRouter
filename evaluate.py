import argparse
import csv
import time
from pathlib import Path

from ablation import run_ablation
from configs import MODEL_NAME
from detector import check_groundedness
from generation import STRATEGY_FUNCTIONS
from retrieval import retrieval_quality, retrieve
from router import answer_query, answer_query_with_strategy, choose_strategy, estimate_hallucination_risk
from scoring import score_answer


def load_benchmark(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def route_benchmark(path: str, limit: int | None = None) -> list[dict]:
    rows = load_benchmark(path)

    if limit is not None:
        rows = rows[:limit]

    routed = []

    for row in rows:
        results = retrieve(row["question"])
        quality = retrieval_quality(results, row["question"])
        decision = choose_strategy(row["question"], quality["score"])
        routed.append({
            "question_id": row["question_id"],
            "domain": row["domain"],
            "difficulty": row["difficulty"],
            "reasoning_required": row["reasoning_required"],
            "answer_type": row["answer_type"],
            "hallucination_risk": estimate_hallucination_risk(row["question"]),
            "retrieval_quality": quality["score"],
            "combined_score": decision.combined_score,
            "strategy": decision.strategy,
            "use_retrieval": decision.use_retrieval,
            "reason": decision.reason,
        })

    return routed


def run_answer_subset(
    path: str,
    limit: int,
    models: list[str] | None = None,
    strategy_override: str | None = None,
) -> list[dict]:

    rows = load_benchmark(path)[:limit]
    models = models or [MODEL_NAME]
    outputs = []

    for model in models:
        for row in rows:
            start = time.perf_counter()

            try:
                if strategy_override:
                    result = answer_query_with_strategy(row["question"], strategy_override, model=model)
                else:
                    result = answer_query(row["question"], model=model)
            except Exception as error:
                outputs.append({
                    "question_id": row["question_id"],
                    "domain": row["domain"],
                    "difficulty": row["difficulty"],
                    "model": model,
                    "strategy": strategy_override or "",
                    "hallucination_risk": "",
                    "retrieval_quality": "",
                    "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                    "input_tokens": "",
                    "output_tokens": "",
                    "total_tokens": "",
                    "answer_stability": "",
                    "is_correct": False,
                    "groundedness_rate": "",
                    "flagged_claims": "",
                    "answer": f"ERROR: {error}",
                    "ground_truth_answer": row["ground_truth_answer"],
                })
                continue

            latency_ms = round((time.perf_counter() - start) * 1000, 1)

            generation = result["generation"]
            is_correct = score_answer(generation["answer"], row["ground_truth_answer"], row["answer_type"])
            grounding = check_groundedness(generation["answer"], result["contexts"])

            outputs.append({
                "question_id": row["question_id"],
                "domain": row["domain"],
                "difficulty": row["difficulty"],
                "model": model,
                "strategy": result["decision"]["strategy"],
                "hallucination_risk": result["decision"]["hallucination_risk"],
                "retrieval_quality": result["decision"]["retrieval_quality"],
                "latency_ms": latency_ms,
                "input_tokens": generation.get("input_tokens"),
                "output_tokens": generation.get("output_tokens"),
                "total_tokens": generation.get("total_tokens"),
                "answer_stability": generation.get("answer_stability"),
                "is_correct": is_correct,
                "groundedness_rate": grounding["groundedness_rate"],
                "flagged_claims": grounding["flagged_count"],
                "answer": generation["answer"],
                "ground_truth_answer": row["ground_truth_answer"],
            })

    return outputs


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="TechReasonBench-100.csv")
    parser.add_argument("--output", default="outputs/routing_decisions.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--strategy", choices=list(STRATEGY_FUNCTIONS.keys()))
    parser.add_argument("--models", type=str)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    if args.ablation:
        rows = load_benchmark(args.benchmark)
        if args.limit is not None:
            rows = rows[:args.limit]
        result_rows = run_ablation(rows)
    elif args.generate:
        models = [m.strip() for m in args.models.split(",")] if args.models else None
        result_rows = run_answer_subset(args.benchmark, args.limit or 5, models=models, strategy_override=args.strategy)
    else:
        result_rows = route_benchmark(args.benchmark, args.limit)

    write_csv(args.output, result_rows)
    print(f"Wrote {len(result_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
