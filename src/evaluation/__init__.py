"""Evaluation package for RAG metrics."""

from src.evaluation.metrics import calculate_hit_rate, calculate_mrr
from src.evaluation.golden_dataset import GoldenDataset, EvaluationScenario
from src.evaluation.runner import EvaluationRunner

__all__ = [
    "calculate_hit_rate",
    "calculate_mrr",
    "GoldenDataset",
    "EvaluationScenario",
    "EvaluationRunner",
]
