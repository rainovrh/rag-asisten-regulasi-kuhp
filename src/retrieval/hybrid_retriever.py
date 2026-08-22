"""Hybrid retrieval combining sparse and dense search with RRF."""

from typing import Optional

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import ReRanker
from src.utils.logger import get_logger
from src.config.constants import DEFAULT_RRF_K, DEFAULT_CRAG_THRESHOLD

logger = get_logger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining BM25 and dense search with RRF fusion."""
    
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = DEFAULT_RRF_K,
        crag_threshold: float = DEFAULT_CRAG_THRESHOLD,
        use_bm25: bool = True,
        use_dense: bool = True,
    ) -> None:
        """Initialize hybrid retriever.
        
        Args:
            bm25_retriever: BM25 sparse retriever instance.
            dense_retriever: Dense retriever instance.
            rrf_k: RRF constant for rank fusion.
            crag_threshold: Minimum RRF score to keep a result.
            use_bm25: Whether to use BM25 retriever.
            use_dense: Whether to use dense retriever.
        """
        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        self._rrf_k = rrf_k
        self._reranker = ReRanker(threshold=crag_threshold)
        self._use_bm25 = use_bm25
        self._use_dense = use_dense
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        apply_crag: bool = True,
    ) -> list[tuple[str, float, str]]:
        """Execute search with configured retrievers.
        
        Args:
            query: Search query.
            top_k: Number of final results.
            apply_crag: Whether to apply CRAG filtering.
        
        Returns:
            List of (doc_id, score, text) tuples.
        """
        # Get results from active retrievers
        bm25_results = self._bm25.search(query, top_k=top_k) if self._use_bm25 else []
        dense_results = self._dense.search(query, top_k=top_k) if self._use_dense else []
        
        # Fuse with RRF
        rrf_scores = self._reciprocal_rank_fusion(bm25_results, dense_results)
        
        # Apply CRAG filtering if requested
        if apply_crag:
            filtered_scores = self._reranker.filter(rrf_scores)
        else:
            filtered_scores = rrf_scores
        
        # Get top-k results
        sorted_results = sorted(
            filtered_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        
        # Format results
        results = []
        for doc_id, score in sorted_results:
            try:
                text = self._bm25.get_document(doc_id)
                results.append((doc_id, score, text))
            except KeyError:
                logger.warning(f"Document {doc_id} not found in corpus")
                continue
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[tuple[str, float]],
        dense_results: list[tuple[str, float]],
    ) -> dict[str, float]:
        """Combine results using Reciprocal Rank Fusion.
        
        Args:
            bm25_results: BM25 search results.
            dense_results: Dense search results.
        
        Returns:
            Dictionary of doc_id -> RRF score.
        """
        rrf_scores: dict[str, float] = {}
        
        # Process BM25 results
        if self._use_bm25:
            for rank, (doc_id, _) in enumerate(bm25_results):
                if doc_id:
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
        
        # Process dense results
        if self._use_dense:
            for rank, (doc_id, _) in enumerate(dense_results):
                if doc_id:
                    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
        
        return rrf_scores
