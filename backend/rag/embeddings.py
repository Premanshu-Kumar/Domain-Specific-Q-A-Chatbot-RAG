"""
Embedding & Vector Store Module
================================
Generates dense vector embeddings with Sentence Transformers and
manages a ChromaDB collection for persistent storage and similarity search.

The embedding model (default: ``all-MiniLM-L6-v2``) runs locally —
no external API key is required for indexing or retrieval.
"""

import logging
from typing import Dict, Any, List

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Generates embeddings and manages the ChromaDB vector store.

    Responsibilities:
        • Load and cache the SentenceTransformer model
        • Create / open a persistent ChromaDB collection
        • Index document chunks (embed + store)
        • Similarity search over stored chunks
        • Collection statistics & document deletion
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./vectorstore",
        collection_name: str = "rag_documents",
    ):
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # ── Load embedding model ────────────────────────────
        logger.info("Loading embedding model: %s …", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded  (dim=%d)", self.model.get_embedding_dimension())

        # ── Initialize ChromaDB ─────────────────────────────
        logger.info("Opening ChromaDB at: %s", persist_directory)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' ready — %d vectors stored",
            collection_name,
            self.collection.count(),
        )

    # ── Embedding generation ─────────────────────────────────

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a batch of texts into dense vectors.

        Args:
            texts: Plain-text strings to embed.

        Returns:
            List of float vectors (one per input text).
        """
        logger.info("Generating embeddings for %d text(s) …", len(texts))
        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 50,
            batch_size=64,
        )
        return embeddings.tolist()

    # ── Indexing ─────────────────────────────────────────────

    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
    ) -> Dict[str, Any]:
        """
        Embed and index a list of document chunks.

        Args:
            chunks:      Each dict must have ``text`` and ``metadata`` keys.
            document_id: Unique ID for the source document (used as ID prefix).

        Returns:
            Result dict with ``success``, ``chunks_added``, and ``total_chunks``.
        """
        if not chunks:
            return {"success": False, "error": "No chunks to index", "chunks_added": 0}

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"{document_id}__chunk_{i}" for i in range(len(chunks))]

        # Ensure all metadata values are ChromaDB-compatible (str, int, float, bool)
        sanitized_meta = [self._sanitize_metadata(m) for m in metadatas]

        embeddings = self.generate_embeddings(texts)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=sanitized_meta,
        )

        total = self.collection.count()
        logger.info(
            "Indexed %d chunks (document=%s). Collection total: %d",
            len(chunks),
            document_id,
            total,
        )
        return {
            "success": True,
            "chunks_added": len(chunks),
            "total_chunks": total,
            "document_id": document_id,
        }

    # ── Search ───────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: Natural-language question or search string.
            top_k: Maximum number of results.

        Returns:
            List of result dicts with ``text``, ``metadata``,
            ``distance`` (cosine), and ``similarity`` (1 − distance).
        """
        if self.collection.count() == 0:
            logger.warning("Search called on empty collection")
            return []

        query_embedding = self.model.encode([query]).tolist()

        n = min(top_k, self.collection.count())
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                dist = results["distances"][0][i] if results["distances"] else None
                hits.append(
                    {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": dist,
                        "similarity": round(1 - dist, 4) if dist is not None else None,
                    }
                )

        logger.info("Search returned %d results for query: '%.60s…'", len(hits), query)
        return hits

    # ── Collection management ────────────────────────────────

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return summary statistics for the active collection."""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "embedding_model": self.model_name,
            "embedding_dimension": self.model.get_embedding_dimension(),
            "persist_directory": self.persist_directory,
        }

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Remove all chunks that belong to *document_id*.

        Matching is done by ID prefix (``<document_id>__chunk_``).
        """
        try:
            # Fetch all IDs in the collection and filter by prefix
            all_data = self.collection.get()
            matching_ids = [
                cid for cid in all_data["ids"] if cid.startswith(f"{document_id}__")
            ]
            if matching_ids:
                self.collection.delete(ids=matching_ids)
                logger.info("Deleted %d chunks for document %s", len(matching_ids), document_id)
            return {"success": True, "deleted_count": len(matching_ids)}
        except Exception as exc:
            logger.error("Failed to delete document %s: %s", document_id, exc)
            return {"success": False, "error": str(exc), "deleted_count": 0}

    def reset_collection(self) -> Dict[str, Any]:
        """Delete and recreate the collection (use with caution)."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Collection '%s' has been reset", self.collection_name)
            return {"success": True, "message": "Collection reset"}
        except Exception as exc:
            logger.error("Failed to reset collection: %s", exc)
            return {"success": False, "error": str(exc)}

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure every metadata value is a ChromaDB-compatible primitive
        (str | int | float | bool).  Non-primitive values are cast to str.
        """
        sanitized = {}
        for key, value in meta.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized
