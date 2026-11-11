"""Hybrid retrieval combining sparse (BM25) and dense (FAISS) search with Reciprocal Rank Fusion (Blok 2, Kotak 8 s.d. 13)."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple

from src.config.constants import DEFAULT_CRAG_THRESHOLD, DEFAULT_FINAL_TOP_K, DEFAULT_RRF_K
from src.preprocessing.normalizer import LegalQueryNormalizer, UserQueryValidator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.reranker import ReRanker
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedLegalChunk:
    """Rich structured container for Top-K re-ranked legal chunk (Kotak 13).
    
    Supports backward-compatible unpacking:
        chunk_id, score, text = retrieved_chunk
    """

    chunk_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: int = 1
    pasal: str = ""
    bab: str = ""
    buku: str = ""
    ayat: Optional[Any] = None
    penjelasan: Optional[str] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None

    def __iter__(self):
        """Enable tuple unpacking for legacy compatibility: (chunk_id, score, text)."""
        return iter((self.chunk_id, self.score, self.text))

    def __getitem__(self, item: int):
        """Enable index subscripting: chunk[0], chunk[1], chunk[2]."""
        return (self.chunk_id, self.score, self.text)[item]

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into structured dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "rrf_score": self.score,
            "pasal": self.pasal,
            "bab": self.bab,
            "buku": self.buku,
            "ayat": self.ayat,
            "penjelasan": self.penjelasan,
            "text": self.text,
            "bm25_score": self.bm25_score,
            "bm25_rank": self.bm25_rank,
            "dense_score": self.dense_score,
            "dense_rank": self.dense_rank,
            "metadata": self.metadata,
        }


class HybridRetriever:
    """Hybrid retrieval combining BM25 and dense FAISS search with RRF fusion (Blok 2).
    
    Pipeline Steps:
    - Kotak 8 : User Legal Case Inquiry (Boundary validation, malicious input rejection)
    - Kotak 9 : Query Normalization (Abbreviation expansion, case folding, keyword retention)
    - Kotak 10: BM25 Lexical Search Top-K
    - Kotak 11: FAISS Semantic Search Top-K
    - Kotak 12: Reciprocal Rank Fusion (k=60) & Monotonic Score Ordering
    - Kotak 13: Top-K Re-ranked Legal Chunks output
    """

    def __init__(
        self,
        bm25_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        rrf_k: int = DEFAULT_RRF_K,
        crag_threshold: float = 0.0,
        use_bm25: bool = True,
        use_dense: bool = True,
        candidate_k: int = 30,
        w_bm25: float = 2.0,
        w_dense: float = 0.3,
        bm25_primacy_threshold: float = 12.0,
        bm25_primacy_ratio: float = 1.05,
    ) -> None:
        """Initialize hybrid retriever with BM25 and FAISS dense search.

        Args:
            bm25_retriever: BM25 sparse retriever instance.
            dense_retriever: Dense FAISS retriever instance.
            rrf_k: RRF smoothing constant (default: 60).
            crag_threshold: Minimum RRF score threshold for filtering (default: 0.0).
            use_bm25: Whether BM25 lexical search is active.
            use_dense: Whether FAISS semantic search is active.
            candidate_k: Number of candidate documents retrieved per branch prior to RRF.
            w_bm25: Weight multiplier for BM25 reciprocal ranks (default: 2.0).
            w_dense: Weight multiplier for FAISS dense reciprocal ranks (default: 0.3).
            bm25_primacy_threshold: BM25 score threshold for lexical primacy guardrail (default: 12.0).
            bm25_primacy_ratio: Margin ratio over second BM25 candidate (default: 1.05).
        """
        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        self._rrf_k = rrf_k
        self._crag_threshold = crag_threshold
        self._use_bm25 = use_bm25 and (bm25_retriever is not None)
        self._use_dense = use_dense and (dense_retriever is not None)
        self._candidate_k = candidate_k
        self._w_bm25 = w_bm25
        self._w_dense = w_dense
        self._bm25_primacy_threshold = bm25_primacy_threshold
        self._bm25_primacy_ratio = bm25_primacy_ratio

        # Kotak 8 & Kotak 9 modules
        self._validator = UserQueryValidator()
        self._normalizer = LegalQueryNormalizer()
        self._reranker = ReRanker(threshold=crag_threshold)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_FINAL_TOP_K,
        apply_crag: bool = False,
    ) -> List[RetrievedLegalChunk]:
        """Execute full hybrid retrieval pipeline (Kotak 8 s.d. 13).

        Args:
            query: Raw user inquiry or legal case.
            top_k: Final number of re-ranked chunks to return (Kotak 13).
            apply_crag: Whether to apply CRAG score threshold filtering.

        Returns:
            List of RetrievedLegalChunk objects sorted strictly descending by RRF score.
        """
        # -------------------------------------------------------------
        # KOTAK 8: User Legal Case Inquiry (Validation & Boundary check)
        # -------------------------------------------------------------
        validation = self._validator.validate(query)
        if not validation.is_valid:
            logger.warning(f"[KOTAK 8] Invalid query rejected: {validation.error_message}")
            raise ValueError(validation.error_message)

        if validation.warning_message:
            logger.info(f"[KOTAK 8] Query warning: {validation.warning_message}")

        # -------------------------------------------------------------
        # KOTAK 9: Query Normalization (< 10 ms execution)
        # -------------------------------------------------------------
        normalized_query = self._normalizer.normalize(validation.sanitized_query)
        if not normalized_query:
            normalized_query = validation.sanitized_query

        # Determine number of candidates for candidate retrieval
        k_candidates = max(top_k * 2, self._candidate_k)

        # -------------------------------------------------------------
        # KOTAK 10: BM25 Lexical Search Top-K
        # -------------------------------------------------------------
        bm25_results: List[Tuple[str, float]] = []
        if self._use_bm25 and self._bm25 is not None:
            bm25_results = self._bm25.search(normalized_query, top_k=k_candidates)

        # -------------------------------------------------------------
        # KOTAK 11: FAISS Semantic Search Top-K
        # -------------------------------------------------------------
        dense_results: List[Tuple[str, float]] = []
        if self._use_dense and self._dense is not None:
            dense_results = self._dense.search(normalized_query, top_k=k_candidates)

        # -------------------------------------------------------------
        # KOTAK 12: Reciprocal Rank Fusion (k=60)
        # -------------------------------------------------------------
        fused_scores, details = self._reciprocal_rank_fusion(bm25_results, dense_results)

        # Optional CRAG filtering
        if apply_crag and self._crag_threshold > 0.0:
            filtered = self._reranker.filter(fused_scores)
            if filtered:
                fused_scores = filtered

        # Strict monotonic descending sort: s1 >= s2 >= s3 ...
        sorted_doc_ids = sorted(
            fused_scores.keys(),
            key=lambda doc_id: fused_scores[doc_id],
            reverse=True,
        )

        # Confidence-Gated Lexical Primacy Guardrail:
        # When BM25 top candidate has high lexical confidence (s1 >= threshold and lead ratio >= ratio),
        # protect the top candidate at rank 1, while preserving strict monotonic score ordering.
        if (
            self._use_bm25
            and self._use_dense
            and bm25_results
            and self._bm25_primacy_threshold > 0.0
        ):
            s1 = bm25_results[0][1]
            s2 = bm25_results[1][1] if len(bm25_results) > 1 else 0.0
            if s1 >= self._bm25_primacy_threshold and (s2 == 0 or (s1 / max(s2, 0.1)) >= self._bm25_primacy_ratio):
                b_top = bm25_results[0][0]
                if b_top in sorted_doc_ids:
                    sorted_doc_ids.remove(b_top)
                sorted_doc_ids.insert(0, b_top)

                # Ensure strict descending score monotonicity: score[0] >= score[1]
                if len(sorted_doc_ids) > 1 and fused_scores[sorted_doc_ids[0]] < fused_scores[sorted_doc_ids[1]]:
                    fused_scores[sorted_doc_ids[0]] = fused_scores[sorted_doc_ids[1]] + 1e-5

        sorted_doc_ids = sorted_doc_ids[:top_k]

        # -------------------------------------------------------------
        # KOTAK 13: Top-K Re-ranked Legal Chunks Output
        # -------------------------------------------------------------
        results: List[RetrievedLegalChunk] = []
        for rank_idx, doc_id in enumerate(sorted_doc_ids, start=1):
            rrf_score = fused_scores[doc_id]
            det = details.get(doc_id, {})

            # Retrieve text and metadata from available retrievers
            text, meta = self._resolve_doc_and_meta(doc_id)

            chunk = RetrievedLegalChunk(
                chunk_id=doc_id,
                score=rrf_score,
                text=text,
                metadata=meta,
                rank=rank_idx,
                pasal=meta.get("pasal", ""),
                bab=meta.get("bab", ""),
                buku=meta.get("buku", ""),
                ayat=meta.get("ayat"),
                penjelasan=meta.get("penjelasan"),
                bm25_score=det.get("bm25_score"),
                bm25_rank=det.get("bm25_rank"),
                dense_score=det.get("dense_score"),
                dense_rank=det.get("dense_rank"),
            )
            results.append(chunk)

        return results

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        w_bm25: Optional[float] = None,
        w_dense: Optional[float] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        """Compute Weighted Reciprocal Rank Fusion scores with k=60.

        Formula:
            RRF_Score(d) = sum_{m in {BM25, Dense}} w_m / (k + r_m(d))
            where r_m(d) is the 1-based rank (r >= 1).

        Returns:
            Tuple of (doc_id -> rrf_score, doc_id -> rank_details).
        """
        rrf_scores: Dict[str, float] = {}
        details: Dict[str, Dict[str, Any]] = {}
        wb = w_bm25 if w_bm25 is not None else self._w_bm25
        wd = w_dense if w_dense is not None else self._w_dense

        # 1. Process BM25 rankings
        if self._use_bm25:
            for rank_0, (doc_id, raw_score) in enumerate(bm25_results):
                rank_1 = rank_0 + 1
                rrf_term = wb / (self._rrf_k + rank_1)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_term

                if doc_id not in details:
                    details[doc_id] = {}
                details[doc_id]["bm25_rank"] = rank_1
                details[doc_id]["bm25_score"] = raw_score

        # 2. Process Dense FAISS rankings
        if self._use_dense:
            for rank_0, (doc_id, raw_score) in enumerate(dense_results):
                rank_1 = rank_0 + 1
                rrf_term = wd / (self._rrf_k + rank_1)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_term

                if doc_id not in details:
                    details[doc_id] = {}
                details[doc_id]["dense_rank"] = rank_1
                details[doc_id]["dense_score"] = raw_score

        return rrf_scores, details

    def _resolve_doc_and_meta(self, doc_id: str) -> Tuple[str, Dict[str, Any]]:
        """Safely fetch document text and metadata from BM25 or Dense retriever."""
        # Try BM25 first
        if self._bm25 is not None:
            try:
                text = self._bm25.get_document(doc_id)
                meta = self._bm25.get_metadata(doc_id)
                return text, meta
            except (KeyError, IndexError):
                pass

        # Try Dense FAISS next
        if self._dense is not None:
            try:
                text = self._dense.get_document(doc_id)
                meta = self._dense.get_metadata(doc_id)
                return text, meta
            except (KeyError, IndexError):
                pass

        logger.warning(f"Metadata for chunk {doc_id} could not be resolved from active stores")
        return "", {}
