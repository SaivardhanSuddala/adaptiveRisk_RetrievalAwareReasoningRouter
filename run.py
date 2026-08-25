import sys

from vector_store import clear_collection, document_count
from loader import load_and_index_documents
from router import answer_query


sys.stdout.reconfigure(encoding="utf-8")

clear_collection()

indexed = load_and_index_documents("docs")

print(f"Indexed {indexed} chunks")
print(f"Documents in DB: {document_count()}")

result = answer_query("What is Retrieval Augmented Generation?")

print("\nRoute:\n")
print(result["decision"])

print("\nAnswer:\n")
print(result["generation"]["answer"])
