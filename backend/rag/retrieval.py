"""
Retrieval Module
================
Provides a thin retrieval interface over the EmbeddingManager's search.

In Phase 1 this is a lightweight wrapper; Phase 2 will add:
  • Query rewriting / expansion
  • Hybrid search (dense + sparse)
  • Re-ranking with a cross-encoder
"""

import logging
from typing import Dict, Any, List, Optional

from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant document chunks for a user query.

    Wraps :class:`EmbeddingManager.search` and formats results
    for downstream consumption (prompt building, API response, etc.).
    """

    def __init__(self, embedding_manager: EmbeddingManager, top_k: int = 5):
        """
        Args:
            embedding_manager: Initialized EmbeddingManager instance.
            top_k:             Default number of chunks to return.
        """
        self.embedding_manager = embedding_manager
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query.

        Args:
            query:          Natural-language question.
            top_k:          Override the default number of results.
            min_similarity: Discard results below this cosine similarity.

        Returns:
            Dict with keys:
                query    – the original query
                results  – list of matched chunks (text, metadata, similarity)
                context  – concatenated text of all matched chunks
                sources  – deduplicated list of source filenames
        """
        k = top_k or self.top_k
        raw_results = self.embedding_manager.search(query, top_k=k)

        # Apply minimum-similarity filter
        filtered = [
            r for r in raw_results
            if r.get("similarity") is not None and r["similarity"] >= min_similarity
        ]

        # Build concatenated context string
        context_parts: List[str] = []
        sources: set[str] = set()

        for hit in filtered:
            context_parts.append(hit["text"])
            source = hit.get("metadata", {}).get("source")
            if source:
                sources.add(source)

        context = "\n\n---\n\n".join(context_parts)

        logger.info(
            "Retrieved %d chunks (of %d raw) for query: '%.60s…'",
            len(filtered),
            len(raw_results),
            query,
        )

        return {
            "query": query,
            "results": filtered,
            "context": context,
            "sources": sorted(sources),
        }
