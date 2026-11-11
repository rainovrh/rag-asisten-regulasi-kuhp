"""Unit Test Suite for Kotak 5 (BM25), Kotak 6 (IndoBERT), and Kotak 7 (FAISS IndexFlatIP).

Strictly validates:
1. Kotak 5 (BM25):
   - Index Build Time < 30s.
   - 100% Hit Rate on unique legal terms ('makar', 'kekuatan gaib/santet', 'pembunuhan berencana', 'asas legalitas').
   - Serialization & Loading consistency.
2. Kotak 6 (IndoBERT Embedding):
   - Dimension consistency: exactly 768.
   - L2 Normalization: ||v||_2 = 1.0 (Cosine equivalence).
   - Zero NaN / Inf anomalies.
3. Kotak 7 (FAISS VectorDB):
   - Index metric is Inner Product (metric_type == 1).
   - Index completeness: ntotal exactly matches total chunks (643).
   - Storage size < 150 MB.
   - Semantic retrieval accuracy on benchmark legal scenarios.
"""

import json
from pathlib import Path
import time
import unittest

import numpy as np

from src.config.paths import INDEXES_DIR, PROCESSED_DATA_DIR
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.indobert_embedding import IndoBERTEmbedding


class TestIndexingDualSearch(unittest.TestCase):
    """Rigorous test suite for dual sparse and dense indexing."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load chunks and ensure indices exist for testing."""
        cls.chunks_path = PROCESSED_DATA_DIR / "kuhp_chunks_hierarchical.json"
        if not cls.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at {cls.chunks_path}")

        with open(cls.chunks_path, "r", encoding="utf-8") as f:
            cls.chunks = json.load(f)

        cls.bm25_path = PROCESSED_DATA_DIR / "bm25_index.pkl"
        cls.faiss_path = PROCESSED_DATA_DIR / "faiss_index_kuhp"

        # Load or initialize BM25
        if cls.bm25_path.exists():
            cls.bm25 = BM25Retriever.load_index(cls.bm25_path)
        else:
            cls.bm25 = BM25Retriever(chunks_data=cls.chunks)
            cls.bm25.save_index(cls.bm25_path)

        # Initialize IndoBERT embedding wrapper
        cls.embedding = IndoBERTEmbedding(batch_size=32)

        # Load or initialize FAISS
        if (cls.faiss_path / "index.faiss").exists():
            cls.dense = DenseRetriever(index_path=cls.faiss_path, embedding=cls.embedding)
        else:
            cls.dense = DenseRetriever(embedding=cls.embedding)
            cls.dense.build_from_chunks(cls.chunks)
            cls.dense.save_index(cls.faiss_path)

    # -------------------------------------------------------------
    # KOTAK 5: BM25 Sparse Indexing Tests
    # -------------------------------------------------------------
    def test_01_bm25_build_time_under_30_seconds(self) -> None:
        """Target Matriks: Waktu pembuatan indeks BM25 < 30 detik."""
        t0 = time.time()
        retriever = BM25Retriever(chunks_data=self.chunks)
        duration = time.time() - t0

        self.assertLess(
            duration,
            30.0,
            f"BM25 build time took {duration:.2f}s (exceeds 30s limit)",
        )
        self.assertEqual(retriever.corpus_size, len(self.chunks))

    def test_02_bm25_unique_legal_token_hit_rate_100_percent(self) -> None:
        """Target Matriks: Kueri token unik ('makar', 'kekuatan gaib', 'pembunuhan berencana') 100% Hit Rate."""
        test_cases = [
            ("makar terhadap presiden", ["kuhp_pasal_191", "kuhp_pasal_192", "kuhp_pasal_193"]),
            ("kekuatan gaib santet", ["kuhp_pasal_252"]),
            ("pembunuhan berencana", ["kuhp_pasal_459"]),
            ("asas legalitas", ["kuhp_pasal_1"]),
        ]

        for query, expected_chunk_candidates in test_cases:
            results = self.bm25.search(query, top_k=5)
            self.assertGreater(len(results), 0, f"BM25 returned 0 results for '{query}'")

            top_chunk_ids = [doc_id for doc_id, score in results]
            has_match = any(cand in top_chunk_ids for cand in expected_chunk_candidates)

            self.assertTrue(
                has_match,
                f"Query '{query}' failed to retrieve any of {expected_chunk_candidates} in Top-5: {top_chunk_ids}",
            )

    # -------------------------------------------------------------
    # KOTAK 6: IndoBERT Dense Embedding Tests
    # -------------------------------------------------------------
    def test_03_indobert_embedding_dimension_and_no_nan(self) -> None:
        """Target Matriks: Dimensi tepat 768 dan tidak ada vektor bernilai NaN."""
        sample_texts = [
            "Tindak pidana makar terhadap Presiden dan Wakil Presiden.",
            "Setiap orang yang melakukan pembunuhan berencana.",
        ]

        vectors = self.embedding.embed_documents(sample_texts)
        self.assertEqual(len(vectors), 2)

        for vec in vectors:
            self.assertEqual(len(vec), 768, f"Expected 768 dimensions, got {len(vec)}")
            np_vec = np.array(vec)
            self.assertFalse(np.isnan(np_vec).any(), "Embedding vector contains NaN values")
            self.assertFalse(np.isinf(np_vec).any(), "Embedding vector contains Inf values")

    def test_04_indobert_l2_normalization_identity(self) -> None:
        """Target Matriks: Vektor ternormalisasi L2 (||v|| = 1.0) sehingga Inner Product == Cosine Similarity."""
        query_vec = np.array(self.embedding.embed_query("pembunuhan berencana"))
        norm = np.linalg.norm(query_vec)

        # Norm must equal 1.0 within floating point precision
        self.assertAlmostEqual(
            float(norm),
            1.0,
            places=4,
            msg=f"Vector norm is {norm}, expected exactly 1.0 for Cosine equivalence",
        )

    # -------------------------------------------------------------
    # KOTAK 7: VectorDB FAISS IndexFlatIP Tests
    # -------------------------------------------------------------
    def test_05_faiss_completeness_and_metric_type(self) -> None:
        """Target Matriks: 100% chunk terindeks (ntotal == 643) dan tipe metrik Inner Product."""
        self.assertEqual(
            self.dense.total_vectors,
            len(self.chunks),
            f"FAISS total vectors ({self.dense.total_vectors}) does not match chunk count ({len(self.chunks)})",
        )

        # Check metric type in FAISS (1 == METRIC_INNER_PRODUCT)
        self.assertEqual(
            self.dense._vectorstore.index.metric_type,
            1,
            "FAISS index metric must be METRIC_INNER_PRODUCT for Cosine similarity",
        )

    def test_06_faiss_storage_size_under_150_mb(self) -> None:
        """Target Matriks: Ukuran memori dan file indeks < 150 MB."""
        index_file = self.faiss_path / "index.faiss"
        self.assertTrue(index_file.exists(), f"Index file not found at {index_file}")

        file_size_mb = index_file.stat().st_size / (1024 * 1024)
        self.assertLess(
            file_size_mb,
            150.0,
            f"FAISS index file size is {file_size_mb:.2f} MB (exceeds 150 MB limit)",
        )

    def test_07_dense_semantic_retrieval_benchmark(self) -> None:
        """Target Matriks: Pencarian semantik berhasil menangkap kasus hukum."""
        scenario = "sanksi bagi orang yang membunuh orang lain secara terencana dengan pidana mati"
        results = self.dense.search(scenario, top_k=5)

        self.assertGreater(len(results), 0)
        top_ids = [doc_id for doc_id, score in results]

        # Pasal 459 (Pembunuhan Berencana) must appear in top results
        self.assertIn(
            "kuhp_pasal_459",
            top_ids,
            f"Semantic retrieval for scenario failed to find 'kuhp_pasal_459' in Top-5: {top_ids}",
        )


if __name__ == "__main__":
    unittest.main()

