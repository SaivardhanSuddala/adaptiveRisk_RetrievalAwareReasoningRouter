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