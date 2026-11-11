"""Unit Test Suite for Blok 2: Representation & Indexing (Kueri & Retrieval).

Validates granular processes Kotak 8 through Kotak 13:
- Kotak 8 : User Legal Case Inquiry (Boundary validation, malicious input rejection, zero crash).
- Kotak 9 : Query Normalization (Latency < 10 ms, abbreviation expansion, 100% keyword retention).
- Kotak 10: BM25 Lexical Search Top-K (< 50 ms latency).
- Kotak 11: FAISS Semantic Search Top-K (Cosine similarity, semantic relevance).
- Kotak 12: Reciprocal Rank Fusion (k=60, mathematical accuracy, score monotonicity, latency < 20 ms).
- Kotak 13: Top-K Re-ranked Legal Chunks (Schema integrity, metadata completeness, tuple unpacking).
"""

import time
import unittest

from src.config.paths import PROCESSED_DATA_DIR
from src.preprocessing.normalizer import LegalQueryNormalizer, UserQueryValidator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedLegalChunk
from src.retrieval.indobert_embedding import IndoBERTEmbedding


class TestRetrievalPipeline(unittest.TestCase):
    """Test suite for Blok 2 (Kotak 8 s.d. 13)."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load persistent indices and initialize retriever instances."""
        bm25_path = PROCESSED_DATA_DIR / "bm25_index.pkl"
        faiss_path = PROCESSED_DATA_DIR / "faiss_index_kuhp"

        if not bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index directory not found at {faiss_path}")

        cls.bm25 = BM25Retriever.load_index(bm25_path)
        cls.embedding = IndoBERTEmbedding(model_name_or_path="models/indobert-base-p1")
        cls.dense = DenseRetriever.load_from_dir(faiss_path, embedding=cls.embedding)

        cls.hybrid = HybridRetriever(
            bm25_retriever=cls.bm25,
            dense_retriever=cls.dense,
            rrf_k=60,
            candidate_k=20,
        )

    def test_01_kotak8_user_inquiry_boundary_and_sanitization(self) -> None:
        """Kotak 8: Test empty, boundary, and malicious inquiry validation."""
        validator = UserQueryValidator()

        # 1. Reject empty and whitespace queries
        empty_res = validator.validate("")
        self.assertFalse(empty_res.is_valid)
        self.assertIn("kosong", empty_res.error_message.lower())

        ws_res = validator.validate("   \n\t   ")
        self.assertFalse(ws_res.is_valid)

        none_res = validator.validate(None)
        self.assertFalse(none_res.is_valid)

        # 2. Reject in HybridRetriever without crash
        with self.assertRaises(ValueError):
            self.hybrid.search("")

        with self.assertRaises(ValueError):
            self.hybrid.search("    ")

        # 3. Excessive word count boundary (> 500 words)
        long_query = "hukum " * 550
        long_res = validator.validate(long_query)
        self.assertTrue(long_res.is_valid)
        self.assertEqual(long_res.word_count, 500)
        self.assertIsNotNone(long_res.warning_message)

        # 4. Malicious script and SQL injection sanitization
        malicious_input = "<script>alert('xss')</script> Pasal 1 KUHP; DROP TABLE users; --"
        mal_res = validator.validate(malicious_input)
        self.assertTrue(mal_res.is_valid)
        self.assertNotIn("<script>", mal_res.sanitized_query)
        self.assertNotIn("DROP TABLE", mal_res.sanitized_query)
        self.assertIn("Pasal 1 KUHP", mal_res.sanitized_query)

    def test_02_kotak9_query_normalization_latency_and_retention(self) -> None:
        """Kotak 9: Test normalization latency (< 10 ms) and legal keyword retention."""
        normalizer = LegalQueryNormalizer()

        # 1. Abbreviation expansion
        raw = "UU no 1 tahun 2023 KUHP tentang perbuatan pidana"
        norm = normalizer.normalize(raw)
        self.assertIn("undang-undang", norm)
        self.assertIn("kitab undang-undang hukum pidana", norm)

        # 2. Benchmark latency across multiple legal case queries (< 10 ms)
        sample_queries = [
            "Apakah seseorang bisa dipidana atas perbuatan yang belum diatur UU?",
            "Penetapan adanya tindak pidana dilarang menggunakan analogi hukum.",
            "Tindak pidana santet atau kekuatan gaib menurut Pasal 252 KUHP.",
            "Unsur-unsur pembunuhan berencana pada Pasal 459 KUHP baru.",
            "Bagaimana penerapan hukum yang hidup dalam masyarakat atau customary law?",
        ]

        durations_ms = []
        for q in sample_queries:
            _, dur_ms = normalizer.measure_latency(q)
            durations_ms.append(dur_ms)

        avg_latency_ms = sum(durations_ms) / len(durations_ms)
        max_latency_ms = max(durations_ms)
        print(f"\n[KOTAK 9] Normalization Latency: Avg={avg_latency_ms:.3f} ms, Max={max_latency_ms:.3f} ms")
        self.assertLess(avg_latency_ms, 10.0, f"Average normalization latency {avg_latency_ms} ms exceeds 10 ms")
        self.assertLess(max_latency_ms, 10.0, f"Max normalization latency {max_latency_ms} ms exceeds 10 ms")

        # 3. 100% Keyword retention
        essential_keywords = ["makar", "santet", "pembunuhan", "berencana", "pidana", "denda", "percobaan"]
        test_text = "Tuntutan pidana denda bagi pelaku percobaan makar dan pembunuhan berencana menggunakan santet."
        normalized_test = normalizer.normalize(test_text)
        for kw in essential_keywords:
            self.assertIn(kw, normalized_test, f"Critical legal keyword '{kw}' was accidentally lost during normalization")

    def test_03_kotak10_and_kotak11_retrieval_branches(self) -> None:
        """Kotak 10 & 11: Test sparse BM25 latency (< 50 ms) and dense FAISS semantic scoring."""
        query = "Apakah seseorang bisa dipidana atas perbuatan yang belum diatur undang-undang?"

        # Kotak 10: BM25 Lexical Search
        t0 = time.perf_counter()
        bm25_res = self.bm25.search(query, top_k=20)
        bm25_duration_ms = (time.perf_counter() - t0) * 1000.0
        print(f"\n[KOTAK 10] BM25 Latency: {bm25_duration_ms:.2f} ms (Results: {len(bm25_res)})")
        self.assertLess(bm25_duration_ms, 50.0, f"BM25 latency {bm25_duration_ms:.2f} ms exceeds 50 ms limit")
        self.assertGreater(len(bm25_res), 0)

        # Kotak 11: FAISS Semantic Search
        t0 = time.perf_counter()
        dense_res = self.dense.search(query, top_k=20)
        dense_duration_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[KOTAK 11] FAISS Latency: {dense_duration_ms:.2f} ms (Results: {len(dense_res)})")
        self.assertGreater(len(dense_res), 0)
        # Cosine similarity relevance score should be between -1.0 and 1.0
        for doc_id, score in dense_res:
            self.assertTrue(-1.0 <= score <= 1.0, f"Score {score} out of cosine range [-1, 1]")

    def test_04_kotak12_rrf_mathematical_precision_and_monotonicity(self) -> None:
        """Kotak 12: Verify exact formula RRF_Score(d) = sum 1 / (60 + r(d)), monotonicity, and latency < 20 ms."""
        # 1. Synthetic test for mathematical verification
        mock_bm25 = [("doc_A", 12.5), ("doc_B", 9.2), ("doc_C", 5.1)]
        mock_dense = [("doc_A", 0.88), ("doc_C", 0.75), ("doc_D", 0.62)]

        t0 = time.perf_counter()
        fused_scores, details = self.hybrid._reciprocal_rank_fusion(mock_bm25, mock_dense)
        fusion_duration_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\n[KOTAK 12] RRF Calculation Latency: {fusion_duration_ms:.3f} ms")
        self.assertLess(fusion_duration_ms, 20.0, f"RRF fusion latency {fusion_duration_ms:.2f} ms exceeds 20 ms")

        # Verification of exact weighted RRF values:
        wb = self.hybrid._w_bm25
        wd = self.hybrid._w_dense
        # doc_A: rank 1 in BM25 (wb/(60+1)) + rank 1 in Dense (wd/(60+1))
        expected_doc_a = (wb / 61.0) + (wd / 61.0)
        # doc_B: rank 2 in BM25 (wb/(60+2))
        expected_doc_b = wb / 62.0
        # doc_C: rank 3 in BM25 (wb/(60+3)) + rank 2 in Dense (wd/(60+2))
        expected_doc_c = (wb / 63.0) + (wd / 62.0)
        # doc_D: rank 3 in Dense (wd/(60+3))
        expected_doc_d = wd / 63.0

        self.assertAlmostEqual(fused_scores["doc_A"], expected_doc_a, places=6)
        self.assertAlmostEqual(fused_scores["doc_B"], expected_doc_b, places=6)
        self.assertAlmostEqual(fused_scores["doc_C"], expected_doc_c, places=6)
        self.assertAlmostEqual(fused_scores["doc_D"], expected_doc_d, places=6)

        # 2. Monotonicity check on real search output
        sample_query = "Tindak pidana menyatakan memiliki kekuatan gaib atau santet"
        results = self.hybrid.search(sample_query, top_k=5)
        self.assertGreater(len(results), 1)

        for i in range(len(results) - 1):
            s_current = results[i].score
            s_next = results[i + 1].score
            self.assertGreaterEqual(
                s_current,
                s_next,
                f"Rank monotonicity violated: rank {i+1} ({s_current}) < rank {i+2} ({s_next})",
            )

    def test_05_kotak13_top_k_reranked_chunks_schema_and_integrity(self) -> None:
        """Kotak 13: Test Top-K output schema completeness and unpackability for generator."""
        sample_queries = [
            ("Apakah seseorang bisa dipidana atas perbuatan yang belum diatur undang-undang?", "Pasal 1"),
            ("Sanksi pidana bagi yang menyatakan memiliki kekuatan gaib untuk menimbulkan penyakit", "Pasal 252"),
            ("Unsur pembunuhan berencana", "Pasal 459"),
        ]

        for query, expected_pasal in sample_queries:
            results = self.hybrid.search(query, top_k=5)
            self.assertEqual(len(results), 5)

            # Check top result or top-3 contains target article
            top_pasals = [c.pasal for c in results]
            print(f"\nQuery: '{query}' -> Top Retrieved Pasals: {top_pasals[:3]}")
            self.assertTrue(
                any(expected_pasal in p for p in top_pasals[:3]),
                f"Expected {expected_pasal} not found in top 3 results: {top_pasals}",
            )

            # Validate schema integrity on all retrieved chunks
            for rank_idx, chunk in enumerate(results, start=1):
                self.assertIsInstance(chunk, RetrievedLegalChunk)
                self.assertEqual(chunk.rank, rank_idx)
                self.assertTrue(len(chunk.chunk_id) > 0)
                self.assertTrue(len(chunk.text) > 0)
                self.assertTrue(len(chunk.buku) > 0)
                self.assertTrue(len(chunk.bab) > 0)
                self.assertTrue(len(chunk.pasal) > 0)

                # Test backward-compatible tuple unpacking (chunk_id, score, text)
                cid, score, text = chunk
                self.assertEqual(cid, chunk.chunk_id)
                self.assertEqual(score, chunk.score)
                self.assertEqual(text, chunk.text)

                # Test dictionary conversion
                chunk_dict = chunk.to_dict()
                self.assertIn("chunk_id", chunk_dict)
                self.assertIn("rrf_score", chunk_dict)
                self.assertIn("metadata", chunk_dict)


if __name__ == "__main__":
    unittest.main()

