"""
Text Chunking Module
====================
Cleans raw extracted text and splits it into overlapping chunks
suitable for embedding and retrieval.

Uses LangChain's RecursiveCharacterTextSplitter which tries to keep
semantically related text together by splitting on paragraph, sentence,
and word boundaries in order.
"""

import re
import logging
from typing import Dict, Any, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextChunker:
    """Cleans and splits text into smaller, overlapping chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            chunk_size:    Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between
                           consecutive chunks for context continuity.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            is_separator_regex=False,
        )

    # ── public API ───────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        """
        Normalize and clean raw extracted text.

        Steps:
            1. Remove null bytes and ASCII control characters
            2. Replace tabs with spaces
            3. Collapse runs of horizontal whitespace
            4. Collapse excessive blank lines (>2 → 2)
            5. Strip each line
            6. Strip overall
        """
        # Remove null bytes and control chars (keep \\n, \\r, \\t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Tabs → spaces
        text = text.replace("\t", " ")

        # Collapse horizontal whitespace (preserve newlines)
        text = re.sub(r"[^\S\n]+", " ", text)

        # Collapse 3+ consecutive newlines → 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip each line individually
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Clean text and split it into chunks, each annotated with metadata.

        Args:
            text:     Raw (or pre-cleaned) text to chunk.
            metadata: Base metadata dict attached to every chunk
                      (e.g. source filename, file type).

        Returns:
            List of dicts, each with:
                text     – the chunk content
                metadata – merged base metadata + chunk-level metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking — returning []")
            return []

        cleaned = self.clean_text(text)
        chunks = self._splitter.split_text(cleaned)

        logger.info(
            "Split text (%d chars) into %d chunks  "
            "(chunk_size=%d, overlap=%d)",
            len(cleaned),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )

        base_meta = metadata or {}
        result: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            chunk_meta = {
                **base_meta,
                "chunk_index": idx,
                "chunk_total": len(chunks),
                "chunk_char_count": len(chunk),
            }
            result.append({"text": chunk, "metadata": chunk_meta})

        return result
