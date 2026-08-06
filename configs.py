import os

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

MODEL_NAME = "qwen/qwen3-8b"
TOP_K = 5

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db")


collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"description": "Research RAG document collection with indexing"  }
        
)