import os

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_MODELS = [
    m.strip()
    for m in os.getenv("MISTRAL_MODELS", MISTRAL_MODEL_NAME).split(",")
    if m.strip()
]
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MODELS = [m.strip() for m in os.getenv("GROQ_MODELS", MODEL_NAME).split(",") if m.strip()]

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_MODELS = [m.strip() for m in os.getenv("GEMINI_MODELS", GEMINI_MODEL_NAME).split(",") if m.strip()]

OPENROUTER_MODEL_NAME = os.getenv(
    "OPENROUTER_MODEL",
    "minimax/minimax-m3:free"
)
def get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = 5
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
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


def get_gemini_client() -> OpenAI:
    """Gemini exposes an OpenAI-compatible endpoint, so we reuse the OpenAI SDK."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def get_cerebras_client() -> Cerebras:
    api_key = os.getenv("CEREBRAS_API_KEY")

    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is not set")

    return Cerebras(api_key=api_key)

def get_mistral_client() -> OpenAI:
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
    )


PROVIDER_CLIENT_FACTORIES = {
    "groq": get_groq_client,
    "gemini": get_gemini_client,
    "mistral": get_mistral_client,
    "openrouter": get_openrouter_client,
}


def resolve_model(model: str) -> tuple[str, str]:
    """Split a 'provider/model-name' string into (provider, native_model_name).

    Falls back to groq if no known provider prefix is present, so existing
    unprefixed model names (e.g. MODEL_NAME) keep working unchanged.
    """
    if "/" in model:
        provider, _, name = model.partition("/")
        if provider in PROVIDER_CLIENT_FACTORIES:
            return provider, name

    return "groq", model


def get_client_for_model(model: str):
    provider, native_model = resolve_model(model)
    return PROVIDER_CLIENT_FACTORIES[provider](), native_model


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
            device="cpu",
        )

    return embedding_model
