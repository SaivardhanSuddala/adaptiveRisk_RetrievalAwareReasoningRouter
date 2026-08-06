from embeddings import embed_documents, embed_query

documents = [
    "Transformers use self attention.",
    "RAG retrieves relevant documents.",
    "Python is a programming language."
]

doc_embeddings = embed_documents(documents)
query_embedding = embed_query("What is Retrieval Augmented Generation?")

print("Documents:", len(doc_embeddings))
print("Embedding dimension:", len(doc_embeddings[0]))
print("Query dimension:", len(query_embedding))

print(type(doc_embeddings))
print(type(doc_embeddings[0]))
print(type(query_embedding))