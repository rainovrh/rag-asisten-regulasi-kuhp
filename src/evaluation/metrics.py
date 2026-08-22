"""Evaluation metrics for RAG system."""

import re
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PASAL_PATTERN = re.compile(r'pasal\s+(\d+[a-zA-Z]*)', re.IGNORECASE)


def _get_pasal_chunks(pasal_ref: str, mapping: dict[str, list[str]]) -> set[str]:
    """Get chunk IDs associated with a pasal reference."""
    match = _PASAL_PATTERN.search(pasal_ref)
    if not match:
        return set()
    pasal_key = match.group(1).lower()
    return set(mapping.get(pasal_key, []))


def calculate_hit_rate(
    results: list[list[tuple[str, float]]],
    ground_truths: list[str],
    top_k: int = 5,
    pasal_mapping: Optional[dict[str, list[str]]] = None,
) -> float:
    """Calculate Hit Rate @ K.
    
    Formula:
        Hit@K = (jumlah query yang relevant doc muncul di top-K) / (total query)
    
    Args:
        results: List of retrieval results per query.
        ground_truths: List of expected document IDs per query.
        top_k: K value for Hit@K.
        pasal_mapping: Optional mapping from pasal numbers to chunk IDs.
    
    Returns:
        Hit rate (0-1).
    """
    if len(results) != len(ground_truths):
        raise ValueError("results and ground_truths must have same length")
    
    hits = 0
    for result_list, gt in zip(results, ground_truths):
        top_k_results = [doc_id for doc_id, _ in result_list[:top_k]]
        if pasal_mapping:
            expected_chunks = _get_pasal_chunks(gt, pasal_mapping)
            if expected_chunks:
                hit = any(doc_id in expected_chunks for doc_id in top_k_results)
            else:
                hit = gt in top_k_results
        else:
            hit = gt in top_k_results
        if hit:
            hits += 1
    
    return hits / len(results) if results else 0.0


def calculate_mrr(
    results: list[list[tuple[str, float]]],
    ground_truths: list[str],
    pasal_mapping: Optional[dict[str, list[str]]] = None,
) -> float:
    """Calculate Mean Reciprocal Rank.
    
    Formula:
        RR = 1 / rank_relevan_pertama  (0 jika tidak ditemukan)
        MRR = rata-rata(RR) across semua query
    
    Args:
        results: List of retrieval results per query.
        ground_truths: List of expected document IDs per query.
        pasal_mapping: Optional mapping from pasal numbers to chunk IDs.
    
    Returns:
        MRR score (0-1).
    """
    if len(results) != len(ground_truths):
        raise ValueError("results and ground_truths must have same length")
    
    reciprocal_ranks = []
    for result_list, gt in zip(results, ground_truths):
        expected_chunks = _get_pasal_chunks(gt, pasal_mapping) if pasal_mapping else set()
        for rank, (doc_id, _) in enumerate(result_list, start=1):
            if expected_chunks:
                if doc_id in expected_chunks:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                if doc_id == gt:
                    reciprocal_ranks.append(1.0 / rank)
                    break
        else:
            reciprocal_ranks.append(0.0)
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
