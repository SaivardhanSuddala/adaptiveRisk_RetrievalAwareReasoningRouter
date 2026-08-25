from typing import List

from configs import collection


def add_documents(
    ids: List[str],
    documents: List[str],
    embeddings: List[List[float]],
    metadatas: List[dict],
) -> None:

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def search(
    query_embedding: List[float],
    top_k: int = 5,
) -> dict:

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )


def document_count() -> int:
    return collection.count()


def clear_collection() -> None:
    ids = collection.get()["ids"]

    if ids:
        collection.delete(ids=ids)
