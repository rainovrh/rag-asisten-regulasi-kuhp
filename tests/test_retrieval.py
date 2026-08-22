"""Tests for retrieval module."""

import pytest
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import ReRanker


class TestBM25Retriever:
    """Tests for BM25Retriever."""
    
    def test_search_returns_results(self, sample_corpus: dict) -> None:
        """Test that search returns results."""
        retriever = BM25Retriever(corpus_data=sample_corpus)
        results = retriever.search("keyakinan", top_k=2)
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    def test_search_empty_query_raises(self, sample_corpus: dict) -> None:
        """Test that empty query raises ValueError."""
        retriever = BM25Retriever(corpus_data=sample_corpus)
        with pytest.raises(ValueError):
            retriever.search("", top_k=5)


class TestReRanker:
    """Tests for ReRanker."""
    
    def test_filter_removes_low_scores(self) -> None:
        """Test that filtering removes low scores."""
        reranker = ReRanker(threshold=0.5)
        scores = {"doc1": 0.8, "doc2": 0.3, "doc3": 0.6}
        filtered = reranker.filter(scores)
        assert "doc2" not in filtered
        assert "doc1" in filtered
    
    def test_filter_keeps_all_above_threshold(self) -> None:
        """Test that all scores above threshold are kept."""
        reranker = ReRanker(threshold=0.1)
        scores = {"doc1": 0.8, "doc2": 0.3}
        filtered = reranker.filter(scores)
        assert len(filtered) == 2
