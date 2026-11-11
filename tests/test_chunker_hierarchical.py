"""Validation Test Suite for Hierarchical Semantic Chunking and Metadata Enrichment.

Strictly validates:
1. 100% Schema Conformity (Pydantic LegalChunk and LegalChunkMetadata validation).
2. Boundary Integrity: All 624 articles + Konsiderans represented without omission.
3. 0% Ayat Truncation: No clauses cut mid-sentence; structural integrity preserved.
4. Dynamic Slicing: Composite long articles (Pasal 622) sliced at Ayat level with inherited breadcrumbs.
5. Tri-Block Formatting: Proper separation of [KONTEKS REGULASI], [BUNYI PASAL ASLI], and [PENJELASAN AUTENTIK].
"""

from pathlib import Path
import unittest

from src.config.paths import PROCESSED_DATA_DIR
from src.preprocessing.hierarchical_chunker import HierarchicalChunker
from src.preprocessing.schema import LegalChunk, LegalChunkMetadata


class TestHierarchicalChunker(unittest.TestCase):
    """Rigorous test suite for Hierarchical Semantic Chunking and Metadata Enrichment."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run chunker once and share output across tests for fast execution."""
        cls.corpus_path = PROCESSED_DATA_DIR / "kuhp_corpus_complete.json"
        if not cls.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {cls.corpus_path}")

        cls.chunker = HierarchicalChunker(max_token_threshold=500)
        cls.chunks = cls.chunker.process(cls.corpus_path)
        cls.chunk_map = {c.id_unik: c for c in cls.chunks}

    def test_01_schema_conformity_100_percent(self) -> None:
        """Target: 100% schema conformity (seluruh chunk memenuhi skema Pydantic)."""
        self.assertGreater(len(self.chunks), 624)
        for c in self.chunks:
            # Validate top-level LegalChunk instance
            self.assertIsInstance(c, LegalChunk)
            self.assertTrue(len(c.id_unik) > 0)
            self.assertTrue(len(c.isi_teks) > 0)

            # Validate nested LegalChunkMetadata
            meta = c.metadata
            self.assertIsInstance(meta, LegalChunkMetadata)
            self.assertEqual(c.id_unik, meta.id_unik)
            self.assertTrue("BUKU" in meta.buku.upper() or meta.buku == "PENDAHULUAN")
            self.assertTrue(meta.bab.startswith("BAB") or meta.bab == "KONSIDERANS DAN ASAS FILOSOFIS")
            self.assertGreaterEqual(meta.char_count, 1)
            self.assertGreaterEqual(meta.token_count, 1)

    def test_02_boundary_integrity_all_624_articles_represented(self) -> None:
        """Target: 100% kelengkapan (seluruh 624 Pasal KUHP Baru terakomodasi dalam chunk)."""
        pasal_found = set()
        for c in self.chunks:
            if c.metadata.nomor_pasal is not None:
                pasal_found.add(c.metadata.nomor_pasal)

        self.assertEqual(
            len(pasal_found),
            624,
            f"Expected 624 unique articles, but found {len(pasal_found)}",
        )
        missing = [i for i in range(1, 625) if i not in pasal_found]
        self.assertEqual(missing, [], f"Missing articles from chunks: {missing}")

        # Check Konsiderans chunk presence
        self.assertIn("kuhp_konsiderans", self.chunk_map)
        konsiderans = self.chunk_map["kuhp_konsiderans"]
        self.assertEqual(konsiderans.metadata.pasal, "Konsiderans")
        self.assertIn("Menimbang", konsiderans.isi_teks)

    def test_03_zero_ayat_truncation_in_landmark_articles(self) -> None:
        """Target: 0% Ayat Truncation (tidak ada ayat yang terpotong di tengah kalimat)."""
        # Pasal 1: Ayat (1) dan (2)
        p1 = self.chunk_map.get("kuhp_pasal_1")
        self.assertIsNotNone(p1)
        self.assertIn("(1)", p1.metadata.ayat)
        self.assertIn("(2)", p1.metadata.ayat)
        self.assertIn("kekuatan peraturan pidana", p1.isi_teks.lower())
        self.assertIn("dilarang digunakan analogi", p1.isi_teks.lower())

        # Pasal 2: Ayat (1), (2), (3)
        p2 = self.chunk_map.get("kuhp_pasal_2")
        self.assertIsNotNone(p2)
        self.assertEqual(p2.metadata.ayat, ["(1)", "(2)", "(3)"])
        self.assertIn("hukum yang hidup dalam masyarakat", p2.isi_teks.lower())

        # Pasal 459: Pembunuhan Berencana
        p459 = self.chunk_map.get("kuhp_pasal_459")
        self.assertIsNotNone(p459)
        self.assertIn("pembunuhan berencana", p459.isi_teks.lower())
        self.assertIn("pidana mati", p459.isi_teks.lower())

    def test_04_dynamic_slicing_and_breadcrumb_pasal_622(self) -> None:
        """Target: Pasal komposit raksasa (>500 token) terpecah rapi per-ayat dengan breadcrumb utuh."""
        p622_chunks = [c for c in self.chunks if c.metadata.nomor_pasal == 622]
        # Pasal 622 has 20 ayat; should be sliced into >= 10 ayat chunks
        self.assertGreaterEqual(
            len(p622_chunks),
            15,
            f"Expected Pasal 622 to be sliced into multiple chunks, but got {len(p622_chunks)}",
        )

        # Verify breadcrumb inheritance on all sliced chunks
        for c in p622_chunks:
            self.assertIn("BUKU KEDUA", c.metadata.buku)
            self.assertIn("BAB XXXVII", c.metadata.bab)
            self.assertEqual(c.metadata.pasal, "Pasal 622")
            self.assertIn("### [KONTEKS REGULASI]", c.isi_teks)
            self.assertIn("### [BUNYI PASAL ASLI]", c.isi_teks)

    def test_05_tri_block_formatting_and_penjelasan_handling(self) -> None:
        """Target: Format Tri-Blok benar dan Penjelasan 'Cukup jelas' diabaikan secara efisien."""
        # Pasal 1: Memiliki penjelasan substantif asas legalitas
        p1 = self.chunk_map.get("kuhp_pasal_1")
        self.assertTrue(p1.metadata.has_penjelasan)
        self.assertIsNotNone(p1.metadata.penjelasan)
        self.assertIn("### [KONTEKS REGULASI]", p1.isi_teks)
        self.assertIn("### [BUNYI PASAL ASLI]", p1.isi_teks)
        self.assertIn("### [PENJELASAN AUTENTIK PEMBENTUK UU]", p1.isi_teks)
        self.assertIn("asas legalitas", p1.isi_teks.lower())

        # Pasal 8: Penjelasan 'Cukup jelas' -> Blok Penjelasan ditiadakan
        p8 = self.chunk_map.get("kuhp_pasal_8")
        self.assertFalse(p8.metadata.has_penjelasan)
        self.assertIsNone(p8.metadata.penjelasan)
        self.assertIn("### [KONTEKS REGULASI]", p8.isi_teks)
        self.assertIn("### [BUNYI PASAL ASLI]", p8.isi_teks)
        self.assertNotIn("### [PENJELASAN AUTENTIK PEMBENTUK UU]", p8.isi_teks)


if __name__ == "__main__":
    unittest.main()

