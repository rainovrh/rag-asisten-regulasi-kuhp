"""Sparse retrieval using BM25 algorithm."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.utils.logger import get_logger
from src.utils.validators import validate_file_path
from src.config.constants import BM25_K1, BM25_B

logger = get_logger(__name__)


class BM25Retriever:
    """BM25 sparse retrieval for legal documents."""
    
    def __init__(
        self,
        corpus_path: Optional[Path] = None,
        corpus_data: Optional[dict[str, str]] = None,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        """Initialize BM25 retriever.
        
        Args:
            corpus_path: Path to JSON corpus file.
            corpus_data: Pre-loaded corpus dictionary.
            k1: BM25 k1 parameter (term frequency saturation).
            b: BM25 b parameter (length normalization).
        
        Raises:
            ValueError: If neither corpus_path nor corpus_data is provided.
        """
        if corpus_path is None and corpus_data is None:
            raise ValueError("Either corpus_path or corpus_data must be provided")
        
        self.k1 = k1
        self.b = b
        
        if corpus_data is not None:
            self._corpus = corpus_data
        else:
            corpus_path = validate_file_path(corpus_path)
            logger.info(f"Loading corpus from {corpus_path}")
            with open(corpus_path, "r", encoding="utf-8") as f:
                self._corpus = json.load(f)
        
        self._doc_ids = list(self._corpus.keys())
        self._doc_texts = [str(doc) for doc in self._corpus.values()]
        self._tokenized_corpus = [text.lower().split() for text in self._doc_texts]
        
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info(f"BM25 index built with {len(self._doc_ids)} documents")
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search documents using BM25.
        
        Args:
            query: Search query.
            top_k: Number of top results to return.
        
        Returns:
            List of (doc_id, score) tuples sorted by score descending.
        """
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id = self._doc_ids[idx]
            score = float(scores[idx])
            results.append((doc_id, score))
        
        return results
    
    def get_document(self, doc_id: str) -> str:
        """Get document text by ID.
        
        Args:
            doc_id: Document identifier.
        
        Returns:
            Document text content.
        
        Raises:
            KeyError: If doc_id not found.
        """
        return self._corpus[doc_id]
    
    @property
    def corpus_size(self) -> int:
        """Number of documents in corpus."""
        return len(self._doc_ids)
