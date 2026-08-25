from embeddings import embed_documents, embed_query


documents = [
    "Transformers use self attention.",
    "RAG retrieves relevant documents.",
    "Python is a programming language.",
]

doc_embeddings = embed_documents(documents)
query_embedding = embed_query("What is Retrieval Augmented Generation?")

assert len(doc_embeddings) == 3
assert len(doc_embeddings[0]) == len(query_embedding)
assert isinstance(doc_embeddings, list)
assert isinstance(doc_embeddings[0], list)
assert isinstance(query_embedding, list)

print("Embedding smoke test passed")
