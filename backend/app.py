"""
Flask Application Entry Point
==============================
Creates the Flask app, registers blueprints, and initializes
the RAG indexing pipeline as a shared application-level resource.

Usage:
    python backend/app.py          # development server
    flask --app backend.app run    # alternative
"""

import logging
import os
import sys

# ── Logging setup ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Ensure the project root is on sys.path so imports work
# when running `python backend/app.py` from the project root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask
from flask_cors import CORS

from backend.config import settings
from backend.api.routes import api_bp
from backend.rag.pipeline import IndexingPipeline


def create_app() -> Flask:
    """Application factory."""

    app = Flask(__name__)

    # ── Flask configuration ─────────────────────────────────
    app.config["SECRET_KEY"] = settings.SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = settings.MAX_CONTENT_LENGTH
    app.config["UPLOAD_FOLDER"] = settings.UPLOAD_FOLDER
    app.config["ALLOWED_EXTENSIONS"] = settings.ALLOWED_EXTENSIONS
    app.config["TOP_K"] = settings.TOP_K

    # ── CORS (allow React dev server in Phase 3) ────────────
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ── Ensure directories exist ────────────────────────────
    os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(settings.VECTORSTORE_DIR, exist_ok=True)

    # ── Initialize RAG pipeline ─────────────────────────────
    logger.info("Initializing RAG pipeline …")
    pipeline = IndexingPipeline(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        embedding_model=settings.EMBEDDING_MODEL,
        persist_directory=settings.VECTORSTORE_DIR,
        collection_name=settings.COLLECTION_NAME,
    )
    app.config["PIPELINE"] = pipeline
    logger.info("RAG pipeline initialized ✓")

    # ── Register blueprints ─────────────────────────────────
    app.register_blueprint(api_bp)

    # ── Root route ──────────────────────────────────────────
    @app.route("/")
    def index():
        return {
            "name": "RAG Chatbot API",
            "version": "0.1.0",
            "status": "running",
            "endpoints": {
                "health": "/api/health",
                "upload": "/api/upload  [POST]",
                "search": "/api/search  [POST]",
                "documents": "/api/documents  [GET]",
            },
        }

    logger.info("Flask app created ✓")
    return app


# ── Run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    app = create_app()
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=settings.FLASK_DEBUG,
    )
