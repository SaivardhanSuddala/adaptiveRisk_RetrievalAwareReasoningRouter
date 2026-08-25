import re

from embeddings import embed_query
from vector_store import search


def retrieve(
    query: str,
    top_k: int = 5,
) -> dict:

    query_embedding = embed_query(query)

    return search(
        query_embedding=query_embedding,
        top_k=top_k,
    )


def retrieve_documents(
    query: str,
    top_k: int = 5,
) -> list[str]:

    results = retrieve(
        query=query,
        top_k=top_k,
    )

    return results["documents"][0]


def retrieve_with_scores(
    query: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:

    results = retrieve(
        query=query,
        top_k=top_k,
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    return list(zip(documents, distances))


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _coverage(query: str, documents: list[str]) -> float:
    query_terms = _tokenize(query)

    if not query_terms:
        return 0.0

    covered = set()
    for document in documents:
        covered |= query_terms & _tokenize(document)

    return len(covered) / len(query_terms)


def _redundancy(documents: list[str]) -> float:
    if len(documents) < 2:
        return 0.0

    tokens = [_tokenize(document) for document in documents]
    pairs = [
        (i, j)
        for i in range(len(tokens))
        for j in range(i + 1, len(tokens))
    ]

    if not pairs:
        return 0.0

    return sum(_jaccard(tokens[i], tokens[j]) for i, j in pairs) / len(pairs)


def _diversity(metadatas: list[dict]) -> float:
    if not metadatas:
        return 0.0

    sources = {metadata.get("source") for metadata in metadatas if metadata}

    return len(sources) / len(metadatas)


def retrieval_quality(results: dict, query: str = "") -> dict:
    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []

    if not distances or not documents:
        return {
            "score": 0.0,
            "best_distance": None,
            "margin": 0.0,
            "coverage": 0.0,
            "redundancy": 0.0,
            "diversity": 0.0,
            "available_contexts": 0,
        }

    best_distance = min(distances)
    similarity = max(0.0, min(1.0, 1.0 - best_distance / 2.0))

    margin = 0.0
    if len(distances) > 1:
        rest_mean = sum(distances[1:]) / len(distances[1:])
        margin = max(0.0, min(1.0, (rest_mean - best_distance) / 2.0))

    coverage = _coverage(query, documents)
    redundancy = _redundancy(documents)
    diversity = _diversity(metadatas)

    composite = (
        0.45 * similarity
        + 0.2 * margin
        + 0.2 * coverage
        + 0.15 * diversity
        - 0.1 * redundancy
    )
    score = max(0.0, min(1.0, composite))

    return {
        "score": round(score, 4),
        "best_distance": best_distance,
        "margin": round(margin, 4),
        "coverage": round(coverage, 4),
        "redundancy": round(redundancy, 4),
        "diversity": round(diversity, 4),
        "available_contexts": len(documents),
    }
