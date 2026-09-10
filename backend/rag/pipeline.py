"""
Document Indexing Pipeline
==========================
Orchestrates the end-to-end flow:

    File → Text Extraction → Cleaning → Chunking → Embedding → Vector Store

This is the main entry-point that the API routes call to process
uploaded documents.
"""

import logging
import uuid
from typing import Dict, Any

from .ingestion import DocumentIngestor
from .chunking import TextChunker
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """
    End-to-end document indexing pipeline.

    Usage::

        pipeline = IndexingPipeline(
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="all-MiniLM-L6-v2",
            persist_directory="./vectorstore",
        )
        result = pipeline.process_document("./uploads/handbook.pdf")
        print(result)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./vectorstore",
        collection_name: str = "rag_documents",
    ):
        logger.info("Initializing IndexingPipeline …")
        self.ingestor = DocumentIngestor()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        logger.info("IndexingPipeline ready ✓")

    # ── public API ───────────────────────────────────────────

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Run a document through the full indexing pipeline.

        Steps:
            1. **Extract** — pull raw text from the file
            2. **Chunk**   — clean + split into overlapping segments
            3. **Index**   — embed chunks and store in ChromaDB

        Args:
            file_path: Path to the uploaded document.

        Returns:
            Result dict summarising each stage, including
            ``success``, ``document_id``, chunk counts, etc.
        """
        document_id = uuid.uuid4().hex[:12]
        logger.info("▸ Pipeline start — file='%s'  doc_id='%s'", file_path, document_id)

        # ── Step 1: Text extraction ─────────────────────────
        extraction = self.ingestor.extract_text(file_path)
        if not extraction["success"]:
            return self._result(
                success=False,
                document_id=document_id,
                stage="extraction",
                error=extraction["error"],
            )

        raw_text = extraction["text"]
        metadata = extraction["metadata"]

        if not raw_text.strip():
            return self._result(
                success=False,
                document_id=document_id,
                stage="extraction",
                error="Document contained no extractable text",
            )

        logger.info(
            "  ✓ Extraction — %d chars from '%s'",
            len(raw_text),
            metadata.get("source", "?"),
        )

        # ── Step 2: Chunking ────────────────────────────────
        chunks = self.chunker.chunk_text(raw_text, metadata=metadata)
        if not chunks:
            return self._result(
                success=False,
                document_id=document_id,
                stage="chunking",
                error="Chunking produced zero chunks",
            )

        logger.info("  ✓ Chunking  — %d chunks", len(chunks))

        # ── Step 3: Embedding & indexing ────────────────────
        indexing = self.embedding_manager.add_documents(chunks, document_id)

        logger.info(
            "  ✓ Indexing  — %d chunks indexed  (total in store: %d)",
            indexing.get("chunks_added", 0),
            indexing.get("total_chunks", 0),
        )

        return self._result(
            success=indexing["success"],
            document_id=document_id,
            source=metadata.get("source", "unknown"),
            file_type=metadata.get("file_type", "unknown"),
            text_length=len(raw_text),
            chunks_created=len(chunks),
            chunks_indexed=indexing.get("chunks_added", 0),
            total_in_store=indexing.get("total_chunks", 0),
            error=indexing.get("error"),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return vector-store statistics."""
        return self.embedding_manager.get_collection_stats()

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _result(**kwargs) -> Dict[str, Any]:
        return kwargs
