from vector_store import clear_collection, document_count
from loader import load_and_index_documents
from retrieval import retrieve_documents


clear_collection()

indexed = load_and_index_documents("docs")

print(f"Indexed {indexed} chunks")
print(f"Documents in DB: {document_count()}")

results = retrieve_documents("What is Retrieval Augmented Generation?")

print("\nRetrieved:\n")

for doc in results:
    
    print(doc)
    print("-" * 50)