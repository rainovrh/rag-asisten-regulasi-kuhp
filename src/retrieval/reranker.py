"""Re-ranking and filtering for retrieval results."""

from typing import Optional

from src.utils.logger import get_logger
from src.config.constants import DEFAULT_CRAG_THRESHOLD

logger = get_logger(__name__)


class ReRanker:
    """Re-rank and filter retrieval results."""
    
    def __init__(self, threshold: float = DEFAULT_CRAG_THRESHOLD) -> None:
        """Initialize re-ranker.
        
        Args:
            threshold: Minimum score threshold for CRAG filtering.
        """
        self.threshold = threshold
    
    def filter(self, scores: dict[str, float]) -> dict[str, float]:
        """Filter results below threshold (CRAG).
        
        Args:
            scores: Dictionary of doc_id -> score.
        
        Returns:
            Filtered dictionary.
        """
        filtered = {
            doc_id: score
            for doc_id, score in scores.items()
            if score >= self.threshold
        }
        
        dropped = len(scores) - len(filtered)
        if dropped > 0:
            logger.info(f"CRAG filtered {dropped} low-relevance documents")
        
        return filtered
    
    def rerank(
        self,
        results: list[tuple[str, float]],
        scores: Optional[dict[str, float]] = None,
    ) -> list[tuple[str, float]]:
        """Re-rank results by score.
        
        Args:
            results: List of (doc_id, score) tuples.
            scores: Optional new scores to use for ranking.
        
        Returns:
            Re-ranked results.
        """
        if scores is not None:
            results = [(doc_id, scores.get(doc_id, 0.0)) for doc_id, _ in results]
        
        return sorted(results, key=lambda x: x[1], reverse=True)
