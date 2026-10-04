"""
Phase 2 — RAG Query & Generation Tests
======================================
Verifies retrieval, generation (stub mode), and the full query pipeline.

Run:
    pytest tests/test_phase2.py -v
"""

from __future__ import annotations

import os
import tempfile
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rag.embeddings import EmbeddingManager
from backend.rag.retrieval import Retriever
from backend.rag.generation import Generator
from backend.rag.pipeline import IndexingPipeline


SAMPLE_TEXT = """
Company Leave Policy

Employees are entitled to 20 days of annual leave per calendar year.
Sick leave is provided for up to 10 days per year with a doctor's note.
Maternity leave is 26 weeks as per the Maternity Benefit Act.
Paternity leave is 15 days for the primary caregiver.
Unused annual leave may be carried forward up to a maximum of 5 days.
"""


@pytest.fixture
def vectorstore_dir():
    tmpdir = tempfile.mkdtemp(prefix="rag_p2_vs_")
    yield tmpdir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_txt_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(SAMPLE_TEXT)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def indexed_pipeline(sample_txt_file, vectorstore_dir):
    pipeline = IndexingPipeline(
        chunk_size=300,
        chunk_overlap=50,
        embedding_model="all-MiniLM-L6-v2",
        persist_directory=vectorstore_dir,
        collection_name="test_phase2",
        top_k=3,
        min_similarity=0.0,
        llm_provider="stub",  # force stub — no API key needed
    )
    result = pipeline.process_document(sample_txt_file)
    assert result["success"] is True
    return pipeline


# ═══════════════════════════════════════════════════════════
# Retriever
# ═══════════════════════════════════════════════════════════

class TestRetriever:
    def test_retrieve_returns_context(self, indexed_pipeline):
        retriever = indexed_pipeline.retriever
        result = retriever.retrieve("How many annual leave days?", top_k=3)

        assert result["query"] == "How many annual leave days?"
        assert len(result["results"]) > 0
        assert "20" in result["context"] or "annual leave" in result["context"].lower()
        assert len(result["sources"]) > 0

    def test_retrieve_empty_query(self, indexed_pipeline):
        result = indexed_pipeline.retriever.retrieve("")
        assert result["results"] == []
        assert result["context"] == ""

    def test_min_similarity_filter(self, indexed_pipeline):
        # Very high threshold should filter everything (or almost everything)
        result = indexed_pipeline.retriever.retrieve(
            "quantum physics black holes",
            top_k=5,
            min_similarity=0.95,
        )
        # May or may not return results depending on embedding similarity,
        # but the call itself must succeed
        assert "results" in result
        assert "context" in result


# ═══════════════════════════════════════════════════════════
# Generator (stub mode)
# ═══════════════════════════════════════════════════════════

class TestGenerator:
    def test_stub_generate_with_context(self):
        gen = Generator(provider="stub")
        result = gen.generate(
            query="How many leave days?",
            context="Employees get 20 days of annual leave.",
            sources=["policy.txt"],
        )
        assert "answer" in result
        assert result["provider"] == "stub"
        assert "policy.txt" in result["sources"]
        assert "20 days" in result["answer"] or "Stub mode" in result["answer"]

    def test_stub_generate_empty_context(self):
        gen = Generator(provider="stub")
        result = gen.generate(
            query="Anything?",
            context="",
            sources=[],
        )
        assert "don't have enough information" in result["answer"].lower()
        assert result["sources"] == []


# ═══════════════════════════════════════════════════════════
# Full RAG query
# ═══════════════════════════════════════════════════════════

class TestRAGQuery:
    def test_query_returns_answer(self, indexed_pipeline):
        result = indexed_pipeline.query("How many days of annual leave?")

        assert result["success"] is True
        assert result["answer"]
        assert result["result_count"] > 0
        assert "model" in result
        assert "provider" in result

    def test_query_empty_question(self, indexed_pipeline):
        result = indexed_pipeline.query("")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_query_with_history(self, indexed_pipeline):
        history = [
            {"role": "user", "content": "Tell me about leave policy"},
            {"role": "assistant", "content": "The company provides several types of leave."},
        ]
        result = indexed_pipeline.query(
            "How many annual leave days specifically?",
            conversation_history=history,
        )
        assert result["success"] is True
        assert result["answer"]
