"""Sparse Lexical Retrieval using BM25Okapi tailored for Indonesian Legal Documents."""

import json
from pathlib import Path
import pickle
import re
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from rank_bm25 import BM25Okapi

from src.config.constants import BM25_B, BM25_K1
from src.utils.logger import get_logger
from src.utils.validators import validate_file_path

logger = get_logger(__name__)


def indonesian_legal_tokenizer(text: str) -> List[str]:
    """Tokenize Indonesian legal text preserving compound words and legal operators.

    - Retains hyphenated compound terms (e.g. 'undang-undang', 'terang-terangan').
    - Preserves statutory numbers and article designations ('pasal', '1', 'ayat', '2').
    - Performs case folding.
    - Avoids destructive stemming to protect distinctive legal morphology and statuses.
    """
    cleaned = text.lower()
    # Extract words including hyphenated words and alphanumeric tokens
    tokens = re.findall(r'\b[a-z0-9]+(?:-[a-z0-9]+)*\b', cleaned)
    return tokens


class BM25Retriever:
    """BM25 sparse retriever for KUHP regulatory chunks with fast persistence."""

    def __init__(
        self,
        chunks_path: Optional[Path] = None,
        chunks_data: Optional[List[dict]] = None,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        """Initialize BM25 retriever.

        Args:
            chunks_path: Path to JSON chunks file (e.g. kuhp_chunks_hierarchical.json).
            chunks_data: Pre-loaded list of chunk dictionaries.
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
        """
        self.k1 = k1
        self.b = b

        self._doc_ids: List[str] = []
        self._doc_texts: List[str] = []
        self._doc_metadata: List[dict] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

        if chunks_data is not None:
            self._build_index(chunks_data)
        elif chunks_path is not None:
            chunks_path = validate_file_path(chunks_path)
            logger.info(f"Loading hierarchical chunks from {chunks_path}")
            with open(chunks_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            # Handle list of chunks or legacy dict
            if isinstance(loaded, list):
                self._build_index(loaded)
            elif isinstance(loaded, dict):
                # Legacy dict adaptation
                dict_chunks = [{"id_unik": k, "isi_teks": v, "metadata": {}} for k, v in loaded.items()]
                self._build_index(dict_chunks)

    def _build_index(self, chunks: List[dict]) -> None:
        """Build BM25 index from chunk objects."""
        logger.info(f"Building BM25 sparse index for {len(chunks)} legal chunks")
        self._doc_ids = []
        self._doc_texts = []
        self._doc_metadata = []
        self._tokenized_corpus = []

        for c in chunks:
            doc_id = c.get("id_unik", "")
            text = c.get("isi_teks", "")
            meta = c.get("metadata", {})

            self._doc_ids.append(doc_id)
            self._doc_texts.append(text)
            self._doc_metadata.append(meta)
            self._tokenized_corpus.append(indonesian_legal_tokenizer(text))

        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self.k1, b=self.b)
        logger.info(f"BM25 index successfully built with {len(self._doc_ids)} documents")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search legal chunks using BM25.

        Args:
            query: Search query text.
            top_k: Number of highest-ranked results to return.

        Returns:
            List of (chunk_id, bm25_score) tuples sorted descending by score.
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        if self._bm25 is None:
            raise RuntimeError("BM25 index is not initialized. Build or load an index first.")

        tokenized_query = indonesian_legal_tokenizer(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Tuple[str, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self._doc_ids[idx], score))

        return results

    def get_document(self, doc_id: str) -> str:
        """Get chunk text by unique identifier."""
        try:
            idx = self._doc_ids.index(doc_id)
            return self._doc_texts[idx]
        except ValueError:
            raise KeyError(f"Chunk ID '{doc_id}' not found in BM25 index")

    def get_metadata(self, doc_id: str) -> dict:
        """Get chunk metadata by unique identifier."""
        try:
            idx = self._doc_ids.index(doc_id)
            return self._doc_metadata[idx]
        except ValueError:
            raise KeyError(f"Chunk ID '{doc_id}' not found in BM25 index")

    def save_index(self, output_path: Path) -> None:
        """Persist BM25 index and corpus state to disk via pickle."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "doc_ids": self._doc_ids,
            "doc_texts": self._doc_texts,
            "doc_metadata": self._doc_metadata,
            "tokenized_corpus": self._tokenized_corpus,
            "k1": self.k1,
            "b": self.b,
            "bm25": self._bm25,
        }
        with open(output_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Serialized BM25 index ({len(self._doc_ids)} docs) to {output_path}")

    @classmethod
    def load_index(cls, index_path: Path) -> "BM25Retriever":
        """Load pre-computed BM25 index from disk in milliseconds."""
        index_path = validate_file_path(index_path, must_exist=True)
        logger.info(f"Loading persistent BM25 index from {index_path}")
        with open(index_path, "rb") as f:
            payload = pickle.load(f)

        retriever = cls.__new__(cls)
        retriever.k1 = payload["k1"]
        retriever.b = payload["b"]
        retriever._doc_ids = payload["doc_ids"]
        retriever._doc_texts = payload["doc_texts"]
        retriever._doc_metadata = payload["doc_metadata"]
        retriever._tokenized_corpus = payload["tokenized_corpus"]
        retriever._bm25 = payload["bm25"]
        logger.info(f"Loaded persistent BM25 index with {len(retriever._doc_ids)} documents")
        return retriever

    @property
    def corpus_size(self) -> int:
        """Total number of indexed chunks."""
        return len(self._doc_ids)
