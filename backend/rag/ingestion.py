"""
Document Ingestion Module
=========================
Handles text extraction from supported document formats:
  - PDF  (via pypdf)
  - TXT  (plain text, UTF-8)
  - DOCX (via python-docx)

Each extractor returns raw text that downstream modules (chunking, embedding)
will clean and process further.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Extracts text content from various document formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

    def __init__(self):
        self._extractors = {
            ".pdf": self._extract_from_pdf,
            ".txt": self._extract_from_txt,
            ".docx": self._extract_from_docx,
        }

    # ── public API ───────────────────────────────────────────

    def extract_text(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a document file.

        Args:
            file_path: Absolute or relative path to the document.

        Returns:
            dict with keys:
                text      – extracted raw text
                metadata  – file-level metadata (name, size, type, …)
                success   – bool
                error     – error message or None
        """
        path = Path(file_path)

        if not path.exists():
            return self._fail(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return self._fail(
                f"Unsupported file format: '{ext}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        try:
            text = self._extractors[ext](str(path))
            metadata = {
                "source": path.name,
                "file_path": str(path.resolve()),
                "file_type": ext.lstrip("."),
                "file_size_bytes": path.stat().st_size,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            logger.info(
                "Extracted %d characters from '%s'", len(text), path.name
            )
            return {
                "text": text,
                "metadata": metadata,
                "success": True,
                "error": None,
            }

        except Exception as exc:
            logger.error("Extraction failed for '%s': %s", file_path, exc)
            return self._fail(str(exc))

    # ── private extractors ───────────────────────────────────

    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """Extract text from every page of a PDF."""
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(text)
            else:
                logger.debug("Page %d of '%s' yielded no text", i + 1, file_path)
        return "\n\n".join(pages)

    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        """Read a plain-text file (UTF-8 with fallback)."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    @staticmethod
    def _extract_from_docx(file_path: str) -> str:
        """Extract paragraph text from a Word document."""
        from docx import Document

        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _fail(error: str) -> Dict[str, Any]:
        return {"text": "", "metadata": {}, "success": False, "error": error}
