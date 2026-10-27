import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
VECTOR_DB_DIR = PROJECT_DIR / "chroma_db"
INDEX_MARKER = VECTOR_DB_DIR / ".ready"
COLLECTION_NAME = "beginner_rag_chatbot"

load_dotenv(PROJECT_DIR / ".env")

def require_api_key() -> None:
    """Stop early with a helpful message when the key is missing."""
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise SystemExit(
            "Google API key is missing. Copy .env.example to .env and add your AI Studio key."
        )

def make_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the embedding model used for both indexing and search."""
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    )

def make_vector_store(embeddings: GoogleGenerativeAIEmbeddings) -> Chroma:
    """Open the project's locally persisted Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
    )
