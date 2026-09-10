"""
Answer Generation Module  (Stub — Phase 2)
===========================================
Will integrate an LLM to generate context-aware answers from
retrieved document chunks.

Phase 2 implementation will include:
  • RAG prompt template construction
  • LLM integration (OpenAI / Hugging Face)
  • Hallucination-control system prompt
  • Streaming response support
  • Conversation memory
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class Generator:
    """
    Generates answers using retrieved context + LLM.

    Currently returns a placeholder response; full LLM integration
    will be added in Phase 2.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = ""):
        self.model_name = model_name
        self.api_key = api_key
        logger.info("Generator initialized (model=%s) — stub mode", model_name)

    def generate(
        self,
        query: str,
        context: str,
        sources: List[str],
    ) -> Dict[str, Any]:
        """
        Generate an answer for *query* grounded in *context*.

        Args:
            query:   User's question.
            context: Concatenated relevant text chunks.
            sources: List of source document names.

        Returns:
            Dict with ``answer``, ``sources``, and ``model`` keys.
        """
        # ── Phase 2: replace with real LLM call ─────────────
        logger.info("generate() called — returning stub response")

        if not context.strip():
            answer = (
                "I don't have enough information in my knowledge base to "
                "answer that question. Please upload relevant documents first."
            )
        else:
            answer = (
                f"[Phase 2 — LLM integration pending]\n\n"
                f"Based on {len(sources)} source(s), the following context "
                f"was retrieved for your question:\n\n"
                f"---\n{context[:500]}{'…' if len(context) > 500 else ''}\n---"
            )

        return {
            "answer": answer,
            "sources": sources,
            "model": self.model_name,
        }
