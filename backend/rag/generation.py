"""
Answer Generation Module  (Phase 2)
===================================
Integrates an LLM to generate context-aware, grounded answers from
retrieved document chunks.

Supports:
  • OpenAI (gpt-3.5-turbo, gpt-4o-mini, …)  — primary
  • Groq   (llama-3.1-8b-instant, …)        — fast & free-tier friendly
  • Offline / stub mode when no API key is present

Features:
  • Anti-hallucination system prompt
  • Source-aware context injection
  • Configurable temperature / max_tokens
  • Graceful fallbacks
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Prompt templates ─────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful, accurate domain-specific AI assistant.
Your answers MUST be grounded strictly in the provided context documents.

Rules:
1. Use ONLY the information present in the CONTEXT section below.
2. If the context does not contain enough information to answer the question,
   reply with: "I don't have enough information in the knowledge base to answer that question."
3. Do NOT invent facts, numbers, or details that are not in the context.
4. When possible, quote or paraphrase the relevant parts of the context.
5. Keep answers clear, concise, and professional.
6. At the end of your answer, do NOT list sources yourself — the system will attach them.
"""

USER_PROMPT_TEMPLATE = """CONTEXT:
{context}

---

QUESTION: {question}

Answer the question using only the context above. If the context is insufficient, say so clearly."""


class Generator:
    """
    Generates grounded answers using retrieved context + an LLM.

    Provider selection (priority order):
        1. Explicit ``provider`` argument
        2. Environment / settings (OPENAI_API_KEY → openai, GROQ_API_KEY → groq)
        3. Stub mode (no API key)
    """

    SUPPORTED_PROVIDERS = {"openai", "groq", "stub"}

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: str = "",
        provider: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        groq_api_key: str = "",
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Resolve provider
        if provider == "auto":
            if self.api_key:
                self.provider = "openai"
            elif self.groq_api_key:
                self.provider = "groq"
                # Sensible default for Groq if user left the OpenAI default
                if self.model_name in ("gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"):
                    self.model_name = "llama-3.1-8b-instant"
            else:
                self.provider = "stub"
        else:
            self.provider = provider.lower()
            if self.provider not in self.SUPPORTED_PROVIDERS:
                raise ValueError(
                    f"Unsupported provider '{provider}'. "
                    f"Choose from {self.SUPPORTED_PROVIDERS}"
                )

        self._client = None
        self._init_client()

        logger.info(
            "Generator ready — provider=%s  model=%s",
            self.provider,
            self.model_name,
        )

    # ── Client initialisation ────────────────────────────────

    def _init_client(self) -> None:
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("openai package not installed — falling back to stub")
                self.provider = "stub"
            except Exception as exc:
                logger.error("Failed to init OpenAI client: %s — stub mode", exc)
                self.provider = "stub"

        elif self.provider == "groq":
            try:
                from openai import OpenAI  # Groq is OpenAI-compatible
                self._client = OpenAI(
                    api_key=self.groq_api_key,
                    base_url="https://api.groq.com/openai/v1",
                )
            except ImportError:
                logger.error("openai package not installed — falling back to stub")
                self.provider = "stub"
            except Exception as exc:
                logger.error("Failed to init Groq client: %s — stub mode", exc)
                self.provider = "stub"

    # ── Public API ───────────────────────────────────────────

    def generate(
        self,
        query: str,
        context: str,
        sources: List[str],
        *,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate an answer for *query* grounded in *context*.

        Args:
            query:                User's question.
            context:              Concatenated relevant text chunks.
            sources:              List of source document names.
            conversation_history: Optional list of prior turns
                                  [{"role": "user"|"assistant", "content": "..."}].

        Returns:
            Dict with keys:
                answer, sources, model, provider, tokens_used (if available)
        """
        if not context or not context.strip():
            return {
                "answer": (
                    "I don't have enough information in my knowledge base to "
                    "answer that question. Please upload relevant documents first."
                ),
                "sources": [],
                "model": self.model_name,
                "provider": self.provider,
                "tokens_used": None,
            }

        if self.provider == "stub":
            return self._stub_generate(query, context, sources)

        return self._llm_generate(query, context, sources, conversation_history)

    # ── LLM call ─────────────────────────────────────────────

    def _llm_generate(
        self,
        query: str,
        context: str,
        sources: List[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        user_content = USER_PROMPT_TEMPLATE.format(
            context=context.strip(),
            question=query.strip(),
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # Optional short conversation history (last 4 turns)
        if conversation_history:
            for turn in conversation_history[-4:]:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_content})

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = response.choices[0].message.content.strip()
            tokens = getattr(response.usage, "total_tokens", None)

            logger.info(
                "LLM answer generated (provider=%s, tokens=%s)",
                self.provider,
                tokens,
            )
            return {
                "answer": answer,
                "sources": sources,
                "model": self.model_name,
                "provider": self.provider,
                "tokens_used": tokens,
            }

        except Exception as exc:
            logger.exception("LLM generation failed — returning graceful error")
            return {
                "answer": (
                    "Sorry, I encountered an error while generating the answer. "
                    f"Please try again later. (Details: {type(exc).__name__})"
                ),
                "sources": sources,
                "model": self.model_name,
                "provider": self.provider,
                "tokens_used": None,
                "error": str(exc),
            }

    # ── Stub (no API key) ────────────────────────────────────

    def _stub_generate(
        self,
        query: str,
        context: str,
        sources: List[str],
    ) -> Dict[str, Any]:
        """
        Deterministic offline response used when no LLM API key is configured.
        Still returns the retrieved context so the API is useful for testing.
        """
        logger.info("generate() — stub mode (no LLM API key)")
        preview = context[:800] + ("…" if len(context) > 800 else "")
        answer = (
            "[Stub mode — set OPENAI_API_KEY or GROQ_API_KEY for real answers]\n\n"
            f"Question: {query}\n\n"
            f"Retrieved context from {len(sources)} source(s):\n\n"
            f"---\n{preview}\n---"
        )
        return {
            "answer": answer,
            "sources": sources,
            "model": "stub",
            "provider": "stub",
            "tokens_used": None,
        }
