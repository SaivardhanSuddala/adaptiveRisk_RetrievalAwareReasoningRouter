from typing import List

from configs import embedding_model


def embed_documents(documents: List[str]) -> List[List[float]]:
    return embedding_model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()


def embed_query(query: str) -> List[float]:
    return embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()