"""
Application Configuration
=========================
Centralizes all configuration settings, loaded from environment variables
with sensible defaults for development.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent

UPLOAD_FOLDER = str(BASE_DIR / "uploads")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", str(BASE_DIR / "vectorstore"))

# ──────────────────────────────────────────────
# File Upload
# ──────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# ──────────────────────────────────────────────
# Text Chunking
# ──────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ──────────────────────────────────────────────
# Embedding Model
# ──────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ──────────────────────────────────────────────
# Vector Store (ChromaDB)
# ──────────────────────────────────────────────
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.0"))

# ──────────────────────────────────────────────
# LLM (Phase 2)
# ──────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")  # auto | openai | groq | stub
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ──────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────
FLASK_ENV = os.getenv("FLASK_ENV", "development")
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
