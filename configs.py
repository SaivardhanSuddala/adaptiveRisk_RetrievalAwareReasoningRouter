import os

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MODELS = [m.strip() for m in os.getenv("GROQ_MODELS", MODEL_NAME).split(",") if m.strip()]
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
COLLECTION_NAME = "documents"
SELF_CONSISTENCY_SAMPLES = int(os.getenv("SELF_CONSISTENCY_SAMPLES", "5"))
REACT_MAX_STEPS = int(os.getenv("REACT_MAX_STEPS", "4"))


embedding_model = None

chroma_client = chromadb.PersistentClient(path="./chroma_db")


collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"description": "Research RAG document collection with indexing"},
)


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    return Groq(api_key=api_key)


def get_embedding_model() -> SentenceTransformer:
    global embedding_model

    if embedding_model is None:
        local_files_only = (
            os.getenv("HF_HUB_OFFLINE") == "1"
            or os.getenv("TRANSFORMERS_OFFLINE") == "1"
        )
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            local_files_only=local_files_only,
        )

    return embedding_model
