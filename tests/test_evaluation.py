"""Tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import calculate_hit_rate, calculate_mrr


class TestCalculateHitRate:
    """Tests for calculate_hit_rate."""
    
    def test_perfect_hit_rate(self) -> None:
        """Test perfect hit rate."""
        results = [[("doc1", 1.0)], [("doc2", 1.0)]]
        ground_truths = ["doc1", "doc2"]
        assert calculate_hit_rate(results, ground_truths) == 1.0
    
    def test_zero_hit_rate(self) -> None:
        """Test zero hit rate."""
        results = [[("doc3", 1.0)], [("doc4", 1.0)]]
        ground_truths = ["doc1", "doc2"]
        assert calculate_hit_rate(results, ground_truths) == 0.0
    
    def test_mixed_hit_rate(self) -> None:
        """Test mixed hit rate."""
        results = [[("doc1", 1.0)], [("doc3", 1.0)]]
        ground_truths = ["doc1", "doc2"]
        assert calculate_hit_rate(results, ground_truths) == 0.5


class TestCalculateMRR:
    """Tests for calculate_mrr."""
    
    def test_perfect_mrr(self) -> None:
        """Test perfect MRR."""
        results = [[("doc1", 1.0)], [("doc2", 1.0)]]
        ground_truths = ["doc1", "doc2"]
        assert calculate_mrr(results, ground_truths) == 1.0
    
    def test_zero_mrr(self) -> None:
        """Test zero MRR."""
        results = [[("doc3", 1.0)], [("doc4", 1.0)]]
        ground_truths = ["doc1", "doc2"]
        assert calculate_mrr(results, ground_truths) == 0.0
    
    def test_rank_two_mrr(self) -> None:
        """Test MRR with rank 2."""
        results = [[("doc3", 1.0), ("doc1", 0.5)], [("doc4", 1.0)]]
        ground_truths = ["doc1", "doc2"]
        expected = (1.0 / 2 + 0.0) / 2
        assert calculate_mrr(results, ground_truths) == expected
