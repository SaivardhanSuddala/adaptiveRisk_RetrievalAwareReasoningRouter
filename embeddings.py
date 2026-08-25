from typing import List

from configs import get_embedding_model


def embed_documents(documents: List[str]) -> List[List[float]]:
    embedding_model = get_embedding_model()

    return embedding_model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()


def embed_query(query: str) -> List[float]:
    embedding_model = get_embedding_model()

    return embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()
