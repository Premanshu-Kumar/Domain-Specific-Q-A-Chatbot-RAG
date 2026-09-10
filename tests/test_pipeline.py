"""
Pipeline Integration Tests
===========================
Verifies the end-to-end document indexing pipeline:
    ingestion → chunking → embedding → retrieval

Run:
    pytest tests/test_pipeline.py -v
"""

import os
import tempfile
import pytest

# ── Adjust import path ──────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rag.ingestion import DocumentIngestor
from backend.rag.chunking import TextChunker
from backend.rag.embeddings import EmbeddingManager
from backend.rag.pipeline import IndexingPipeline


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════

SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that combines information
retrieval with generative AI models. Instead of relying solely on the knowledge
encoded in the model's parameters during training, RAG systems first retrieve
relevant documents from an external knowledge base and then use those documents
as additional context when generating a response.

This approach has several advantages:
1. Reduced hallucination — responses are grounded in actual documents.
2. Up-to-date information — the knowledge base can be updated without
   retraining the model.
3. Domain specificity — custom documents can be added for specialised
   use cases.
4. Transparency — sources can be cited alongside the generated answer.

The typical RAG pipeline consists of the following stages:
- Document ingestion and preprocessing
- Text chunking with overlap
- Embedding generation using a dense encoder
- Storage in a vector database
- Query embedding and similarity search
- Context augmentation and LLM response generation
"""


@pytest.fixture
def sample_txt_file():
    """Create a temporary .txt file with sample content."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(SAMPLE_TEXT)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def vectorstore_dir():
    """Create a temporary directory for ChromaDB."""
    tmpdir = tempfile.mkdtemp(prefix="rag_test_vs_")
    yield tmpdir
    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# Tests — Ingestion
# ═══════════════════════════════════════════════════════════

class TestIngestion:
    def test_extract_txt(self, sample_txt_file):
        ingestor = DocumentIngestor()
        result = ingestor.extract_text(sample_txt_file)

        assert result["success"] is True
        assert result["error"] is None
        assert len(result["text"]) > 100
        assert result["metadata"]["file_type"] == "txt"

    def test_extract_nonexistent(self):
        ingestor = DocumentIngestor()
        result = ingestor.extract_text("/nonexistent/file.txt")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_extract_unsupported_format(self, tmp_path):
        unsupported = tmp_path / "data.csv"
        unsupported.write_text("a,b,c")
        ingestor = DocumentIngestor()
        result = ingestor.extract_text(str(unsupported))

        assert result["success"] is False
        assert "unsupported" in result["error"].lower()


# ═══════════════════════════════════════════════════════════
# Tests — Chunking
# ═══════════════════════════════════════════════════════════

class TestChunking:
    def test_clean_text(self):
        chunker = TextChunker()
        dirty = "  Hello \t world  \n\n\n\n  Foo  "
        clean = chunker.clean_text(dirty)

        assert "\t" not in clean
        assert "\n\n\n" not in clean
        assert clean.startswith("Hello")

    def test_chunk_text_produces_chunks(self):
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_text(SAMPLE_TEXT, metadata={"source": "test.txt"})

        assert len(chunks) > 1
        for c in chunks:
            assert "text" in c
            assert "metadata" in c
            assert c["metadata"]["source"] == "test.txt"
            assert "chunk_index" in c["metadata"]

    def test_chunk_empty_text(self):
        chunker = TextChunker()
        chunks = chunker.chunk_text("", metadata={})
        assert chunks == []

    def test_chunk_metadata_enrichment(self):
        chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk_text(SAMPLE_TEXT, metadata={"source": "doc.pdf"})

        for i, c in enumerate(chunks):
            assert c["metadata"]["chunk_index"] == i
            assert c["metadata"]["chunk_total"] == len(chunks)
            assert c["metadata"]["chunk_char_count"] > 0


# ═══════════════════════════════════════════════════════════
# Tests — Embeddings & Vector Store
# ═══════════════════════════════════════════════════════════

class TestEmbeddings:
    def test_generate_embeddings(self, vectorstore_dir):
        mgr = EmbeddingManager(
            model_name="all-MiniLM-L6-v2",
            persist_directory=vectorstore_dir,
            collection_name="test_collection",
        )
        vecs = mgr.generate_embeddings(["Hello world", "Test sentence"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 384  # all-MiniLM-L6-v2 dimension

    def test_add_and_search(self, vectorstore_dir):
        mgr = EmbeddingManager(
            model_name="all-MiniLM-L6-v2",
            persist_directory=vectorstore_dir,
            collection_name="test_search",
        )

        chunks = [
            {"text": "RAG uses retrieval before generation", "metadata": {"source": "a.txt", "chunk_index": 0}},
            {"text": "Python is a programming language", "metadata": {"source": "b.txt", "chunk_index": 0}},
            {"text": "Vector databases store embeddings", "metadata": {"source": "c.txt", "chunk_index": 0}},
        ]
        mgr.add_documents(chunks, document_id="test_doc")

        results = mgr.search("What is retrieval augmented generation?", top_k=2)
        assert len(results) == 2
        # The RAG chunk should rank highest
        assert "retrieval" in results[0]["text"].lower() or "rag" in results[0]["text"].lower()

    def test_collection_stats(self, vectorstore_dir):
        mgr = EmbeddingManager(
            persist_directory=vectorstore_dir,
            collection_name="test_stats",
        )
        stats = mgr.get_collection_stats()
        assert stats["collection_name"] == "test_stats"
        assert stats["total_documents"] == 0


# ═══════════════════════════════════════════════════════════
# Tests — Full Pipeline
# ═══════════════════════════════════════════════════════════

class TestPipeline:
    def test_end_to_end(self, sample_txt_file, vectorstore_dir):
        pipeline = IndexingPipeline(
            chunk_size=300,
            chunk_overlap=50,
            embedding_model="all-MiniLM-L6-v2",
            persist_directory=vectorstore_dir,
            collection_name="test_pipeline",
        )

        result = pipeline.process_document(sample_txt_file)

        assert result["success"] is True
        assert result["chunks_created"] > 0
        assert result["chunks_indexed"] > 0
        assert result["text_length"] > 100

        # Verify search works after indexing
        hits = pipeline.embedding_manager.search("What is RAG?", top_k=3)
        assert len(hits) > 0

    def test_pipeline_stats(self, vectorstore_dir):
        pipeline = IndexingPipeline(
            persist_directory=vectorstore_dir,
            collection_name="test_stats_pipe",
        )
        stats = pipeline.get_stats()
        assert "collection_name" in stats
        assert "total_documents" in stats
