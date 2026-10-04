"""
API Routes
==========
Flask blueprint exposing REST endpoints for the RAG application.

Endpoints:
    POST /api/upload      — Upload and index a document
    GET  /api/documents   — List indexed documents (stats)
    POST /api/search      — Semantic search over indexed chunks
    POST /api/chat        — Full RAG chat with grounded answers + citations
    GET  /api/health      — Health check
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _allowed_file(filename: str) -> bool:
    """Check whether the file extension is in the allowed set."""
    allowed = current_app.config.get("ALLOWED_EXTENSIONS", {"pdf", "txt", "docx"})
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


# ── Health ───────────────────────────────────────────────────

@api_bp.route("/health", methods=["GET"])
def health():
    """Simple liveness probe."""
    return jsonify({"status": "ok", "message": "RAG Chatbot API is running"}), 200


# ── Document Upload ──────────────────────────────────────────

@api_bp.route("/upload", methods=["POST"])
def upload_document():
    """
    Upload a document to be indexed into the knowledge base.

    Expects a multipart/form-data request with a ``file`` field.

    Returns:
        201 on success with indexing summary.
        400 on validation error.
        500 on processing error.
    """
    # Validate file presence
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Use the 'file' form field."}), 400

    file = request.files["file"]

    if file.filename == "" or file.filename is None:
        return jsonify({"error": "No file selected"}), 400

    if not _allowed_file(file.filename):
        allowed = current_app.config.get("ALLOWED_EXTENSIONS", {"pdf", "txt", "docx"})
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(sorted(allowed))}"
        }), 400

    # Save to upload directory
    filename = secure_filename(file.filename)
    upload_dir = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    logger.info("File saved to: %s", file_path)

    # Process through indexing pipeline
    try:
        pipeline = current_app.config["PIPELINE"]
        result = pipeline.process_document(file_path)

        if result.get("success"):
            return jsonify({
                "message": f"Document '{filename}' uploaded and indexed successfully",
                "document_id": result.get("document_id"),
                "source": result.get("source"),
                "file_type": result.get("file_type"),
                "text_length": result.get("text_length"),
                "chunks_created": result.get("chunks_created"),
                "chunks_indexed": result.get("chunks_indexed"),
                "total_in_store": result.get("total_in_store"),
            }), 201
        else:
            return jsonify({
                "error": f"Processing failed at stage: {result.get('stage', 'unknown')}",
                "details": result.get("error"),
            }), 500

    except Exception as exc:
        logger.exception("Upload processing error")
        return jsonify({"error": "Internal processing error", "details": str(exc)}), 500


# ── Search ───────────────────────────────────────────────────

@api_bp.route("/search", methods=["POST"])
def search():
    """
    Semantic search over the indexed knowledge base.

    Expects JSON body::

        { "query": "...", "top_k": 5 }

    Returns:
        200 with matched chunks, context, and sources.
    """
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    top_k = data.get("top_k", current_app.config.get("TOP_K", 5))
    try:
        top_k = int(top_k)
        if top_k < 1 or top_k > 50:
            return jsonify({"error": "top_k must be between 1 and 50"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400

    try:
        pipeline = current_app.config["PIPELINE"]
        results = pipeline.embedding_manager.search(query, top_k=top_k)

        # Build response
        sources = set()
        for r in results:
            src = r.get("metadata", {}).get("source")
            if src:
                sources.add(src)

        return jsonify({
            "query": query,
            "results": results,
            "sources": sorted(sources),
            "result_count": len(results),
        }), 200

    except Exception as exc:
        logger.exception("Search error")
        return jsonify({"error": "Search failed", "details": str(exc)}), 500


# ── Chat (Phase 2) ───────────────────────────────────────────

@api_bp.route("/chat", methods=["POST"])
def chat():
    """
    Full RAG chat endpoint — retrieve relevant chunks and generate a
    grounded answer with source citations.

    Expects JSON body::

        {
          "question": "How many annual leave days are employees entitled to?",
          "top_k": 5,                    # optional
          "min_similarity": 0.25,        # optional
          "history": [                  # optional multi-turn context
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
          ]
        }

    Returns:
        200 with answer, sources, retrieved chunks, model metadata.
        400 on validation error.
        500 on internal error.
    """
    data = request.get_json(silent=True) or {}

    # Accept both "question" and "query" for convenience
    question = (data.get("question") or data.get("query") or "").strip()
    if not question:
        return jsonify({
            "error": "Missing 'question' (or 'query') in request body"
        }), 400

    if len(question) > 4000:
        return jsonify({"error": "Question is too long (max 4000 characters)"}), 400

    # Optional parameters with validation
    top_k = data.get("top_k", current_app.config.get("TOP_K", 5))
    try:
        top_k = int(top_k)
        if top_k < 1 or top_k > 50:
            return jsonify({"error": "top_k must be between 1 and 50"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400

    min_similarity = data.get(
        "min_similarity",
        current_app.config.get("MIN_SIMILARITY", 0.0),
    )
    try:
        min_similarity = float(min_similarity)
        if not 0.0 <= min_similarity <= 1.0:
            return jsonify({"error": "min_similarity must be between 0.0 and 1.0"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "min_similarity must be a number"}), 400

    # Optional conversation history
    history: List[Dict[str, str]] = data.get("history") or data.get("conversation_history") or []
    if history and not isinstance(history, list):
        return jsonify({"error": "'history' must be a list of message objects"}), 400

    # Sanitize history entries
    clean_history: List[Dict[str, str]] = []
    for turn in history[-8:]:  # keep last 8 turns max
        if isinstance(turn, dict):
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                clean_history.append({"role": role, "content": content.strip()[:2000]})

    try:
        pipeline = current_app.config["PIPELINE"]
        result = pipeline.query(
            question=question,
            top_k=top_k,
            min_similarity=min_similarity,
            conversation_history=clean_history or None,
        )

        if not result.get("success"):
            return jsonify({
                "error": result.get("error", "Query failed"),
            }), 400

        response: Dict[str, Any] = {
            "question": result["question"],
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "result_count": result.get("result_count", 0),
            "model": result.get("model"),
            "provider": result.get("provider"),
            "tokens_used": result.get("tokens_used"),
        }

        # Optionally include retrieved chunks (useful for debugging / UI)
        include_chunks = data.get("include_chunks", False)
        if include_chunks:
            response["results"] = result.get("results", [])

        if result.get("error"):
            response["warning"] = result["error"]

        return jsonify(response), 200

    except Exception as exc:
        logger.exception("Chat error")
        return jsonify({"error": "Chat failed", "details": str(exc)}), 500


# ── Collection Stats ─────────────────────────────────────────

@api_bp.route("/documents", methods=["GET"])
def documents_stats():
    """Return vector store collection statistics."""
    try:
        pipeline = current_app.config["PIPELINE"]
        stats = pipeline.get_stats()
        return jsonify(stats), 200
    except Exception as exc:
        logger.exception("Stats error")
        return jsonify({"error": str(exc)}), 500
