"""
Document Indexing & RAG Query Pipeline
======================================
Orchestrates the end-to-end flow:

    File → Text Extraction → Cleaning → Chunking → Embedding → Vector Store

And the query-time RAG flow:

    Query → Retrieve Top-K → Build Context → Generate Grounded Answer

This is the main entry-point that the API routes call.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from .ingestion import DocumentIngestor
from .chunking import TextChunker
from .embeddings import EmbeddingManager
from .retrieval import Retriever
from .generation import Generator

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """
    End-to-end document indexing + RAG query pipeline.

    Usage::

        pipeline = IndexingPipeline(...)
        result = pipeline.process_document("./uploads/handbook.pdf")
        answer = pipeline.query("How many leave days do employees get?")
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./vectorstore",
        collection_name: str = "rag_documents",
        top_k: int = 5,
        min_similarity: float = 0.0,
        # LLM settings
        llm_model: str = "gpt-3.5-turbo",
        openai_api_key: str = "",
        groq_api_key: str = "",
        llm_provider: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
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
        self.retriever = Retriever(
            embedding_manager=self.embedding_manager,
            top_k=top_k,
        )
        self.generator = Generator(
            model_name=llm_model,
            api_key=openai_api_key,
            groq_api_key=groq_api_key,
            provider=llm_provider,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.top_k = top_k
        self.min_similarity = min_similarity
        logger.info("IndexingPipeline ready ✓")

    # ── Document indexing ────────────────────────────────────

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Run a document through the full indexing pipeline.

        Steps:
            1. **Extract** — pull raw text from the file
            2. **Chunk**   — clean + split into overlapping segments
            3. **Index**   — embed chunks and store in ChromaDB
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

    # ── RAG query (Phase 2) ──────────────────────────────────

    def query(
        self,
        question: str,
        *,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG query: retrieve relevant chunks → generate grounded answer.

        Args:
            question:             User's natural-language question.
            top_k:                Override default number of chunks.
            min_similarity:       Override minimum cosine similarity filter.
            conversation_history: Optional prior turns for multi-turn context.

        Returns:
            Dict containing answer, sources, retrieved chunks, model info, etc.
        """
        if not question or not question.strip():
            return {
                "success": False,
                "error": "Question cannot be empty",
                "answer": "",
                "sources": [],
                "results": [],
            }

        k = top_k if top_k is not None else self.top_k
        sim = min_similarity if min_similarity is not None else self.min_similarity

        # 1. Retrieve
        retrieval = self.retriever.retrieve(
            query=question.strip(),
            top_k=k,
            min_similarity=sim,
        )

        # 2. Generate
        generation = self.generator.generate(
            query=question.strip(),
            context=retrieval["context"],
            sources=retrieval["sources"],
            conversation_history=conversation_history,
        )

        return {
            "success": True,
            "question": question.strip(),
            "answer": generation["answer"],
            "sources": generation.get("sources", retrieval["sources"]),
            "results": retrieval["results"],
            "result_count": len(retrieval["results"]),
            "model": generation.get("model"),
            "provider": generation.get("provider"),
            "tokens_used": generation.get("tokens_used"),
            "error": generation.get("error"),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return vector-store statistics."""
        return self.embedding_manager.get_collection_stats()

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _result(**kwargs) -> Dict[str, Any]:
        return kwargs
