"""Zero-Defect Validation Test Suite for KUHP Document Cleaning and Regex Sanitation.

This test suite strictly validates:
1. 100% Extraction Coverage: Exactly 624 articles (Pasal 1 to Pasal 624) with 0 missing.
2. 0% Header/Footer Noise Rate: Total elimination of header artifacts, page numbers, and SK stamps.
3. Konsiderans Extraction: Preserved as standalone chunk with exact metadata {"bagian": "Konsiderans", "jenis": "Pendahuluan"}.
4. Penjelasan Injection: All 624 articles mapped; substantive explanations injected; 'Cukup Jelas' set to None.
5. Ayat Integrity: Zero ayat truncation or OCR parenthesis corruption in landmark articles.
"""

from pathlib import Path
import re
import unittest

from src.config.paths import RAW_DATA_DIR
from src.preprocessing.cleaner import DocumentCleaner, SpatialBounds


class TestCleanerZeroDefect(unittest.TestCase):
    """Rigorous unit test suite for verifying zero-defect preprocessing."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run full extraction once and share across tests for fast execution."""
        cls.pdf_path = RAW_DATA_DIR / "KUHP BARU UU Nomor 1 Tahun 2023.pdf"
        if not cls.pdf_path.exists():
            raise FileNotFoundError(f"KUHP PDF not found at {cls.pdf_path}")

        cls.cleaner = DocumentCleaner()
        cls.result = cls.cleaner.extract_clean_pages(cls.pdf_path)
        cls.articles = cls.result["articles"]
        cls.konsiderans = cls.result["konsiderans"]
        cls.penjelasan_umum = cls.result["penjelasan_umum"]
        cls.metadata = cls.result["metadata"]

    def test_01_all_624_articles_extracted_without_omission(self) -> None:
        """Target: 100% kelengkapan (seluruh 624 Pasal KUHP Baru terakomodasi)."""
        self.assertEqual(
            len(self.articles),
            624,
            f"Expected exactly 624 articles, but found {len(self.articles)}",
        )
        self.assertTrue(
            self.metadata["is_complete"],
            "Metadata indicates missing articles in corpus",
        )
        self.assertEqual(
            self.metadata["missing_articles"],
            [],
            f"Missing articles detected: {self.metadata['missing_articles']}",
        )

        # Check every consecutive number from 1 to 624 exists
        for i in range(1, 625):
            self.assertIn(i, self.articles, f"Pasal {i} is missing from articles dictionary")
            art = self.articles[i]
            self.assertEqual(art["nomor"], i)
            self.assertEqual(art["pasal"], f"Pasal {i}")
            self.assertGreater(
                len(art["isi_teks"]),
                10,
                f"Pasal {i} text is suspiciously empty: '{art['isi_teks']}'",
            )

    def test_02_zero_header_footer_noise_rate(self) -> None:
        """Target: 0 chunk anomali (tidak ada sampah header/footer/SK No)."""
        noise_sk = []
        noise_page_num = []
        noise_salinan = []
        noise_garuda = []

        # Header OCR garbage patterns seen in raw PDF
        garbled_header_pat = re.compile(
            r'(?i)(?:EETITIilIilTI|rEIItrtrf|EITtrELItrEEIIEIn|FRESIDEN\s+REPU|nrrFF\[iriN\])'
        )

        for num, art in self.articles.items():
            text = art["isi_teks"]

            if "SK No" in text:
                noise_sk.append((num, "SK No"))
            if re.search(r'(?m)^-\s*\d+\s*-$', text):
                noise_page_num.append((num, "Page Number Dash"))
            if "SALINAN" in text:
                noise_salinan.append((num, "SALINAN"))
            if garbled_header_pat.search(text):
                noise_garuda.append((num, "Garbled Header"))

        self.assertEqual(noise_sk, [], f"SK No stamps leaked into: {noise_sk}")
        self.assertEqual(noise_page_num, [], f"Page numbers leaked into: {noise_page_num}")
        self.assertEqual(noise_salinan, [], f"SALINAN leaked into: {noise_salinan}")
        self.assertEqual(noise_garuda, [], f"Garbled headers leaked into: {noise_garuda}")

    def test_03_konsiderans_structure_and_metadata(self) -> None:
        """Target: Konsiderans preserved cleanly with required metadata."""
        self.assertIsNotNone(self.konsiderans)
        self.assertEqual(self.konsiderans["bagian"], "Konsiderans")
        self.assertEqual(self.konsiderans["jenis"], "Pendahuluan")

        text = self.konsiderans["isi_teks"]
        self.assertGreater(len(text), 1000, "Konsiderans text is too short")
        self.assertIn("Menimbang", text, "'Menimbang' missing from Konsiderans")
        self.assertIn("Mengingat", text, "'Mengingat' missing from Konsiderans")
        self.assertIn("Pancasila", text, "'Pancasila' missing from Konsiderans")
        self.assertNotIn("SK No", text, "SK No leaked into Konsiderans")
        self.assertNotIn("SALINAN", text, "SALINAN leaked into Konsiderans")

    def test_04_penjelasan_injection_and_cukup_jelas(self) -> None:
        """Target: Penjelasan parsed separately and mapped to corresponding articles."""
        substantive = self.metadata["substantive_penjelasan_count"]
        cukup_jelas = self.metadata["cukup_jelas_penjelasan_count"]

        # KUHP Baru has over 250 articles with substantive explanation and over 300 marked 'Cukup jelas'
        self.assertGreaterEqual(substantive, 250, f"Expected >=250 substantive explanations, got {substantive}")
        self.assertGreaterEqual(cukup_jelas, 300, f"Expected >=300 Cukup Jelas explanations, got {cukup_jelas}")
        self.assertEqual(substantive + cukup_jelas, 624)

        # Check Pasal 1: has substantive explanation on asas legalitas
        penj_1 = self.articles[1]["penjelasan"]
        self.assertIsNotNone(penj_1, "Pasal 1 explanation should not be None")
        self.assertIn("asas legalitas", penj_1.lower(), "Pasal 1 explanation should contain 'asas legalitas'")

        # Check Pasal 2: has substantive explanation on hukum yang hidup dalam masyarakat
        penj_2 = self.articles[2]["penjelasan"]
        self.assertIsNotNone(penj_2, "Pasal 2 explanation should not be None")
        self.assertTrue(
            "hukum yang hidup" in penj_2.lower() or "adat" in penj_2.lower(),
            "Pasal 2 explanation should contain 'hukum yang hidup' or 'adat'",
        )

        # Check Pasal 4: has substantive explanation on wilayah NKRI
        penj_4 = self.articles[4]["penjelasan"]
        self.assertIsNotNone(penj_4, "Pasal 4 explanation should not be None")
        self.assertIn("wilayah negara kesatuan republik indonesia", penj_4.lower())

        # Check Pasal 8 & Pasal 9: purely 'Cukup jelas.' -> must be None as requested
        penj_8 = self.articles[8]["penjelasan"]
        self.assertIsNone(penj_8, f"Pasal 8 is 'Cukup jelas' and should be None, but got: {penj_8}")

        penj_9 = self.articles[9]["penjelasan"]
        self.assertIsNone(penj_9, f"Pasal 9 is 'Cukup jelas' and should be None, but got: {penj_9}")

    def test_05_ayat_integrity_in_sample_articles(self) -> None:
        """Target: Tidak ada ayat yang terpotong di tengah kalimat dan OCR penomoran bersih."""
        # Pasal 1: Ayat (1) dan (2)
        p1 = self.articles[1]["isi_teks"]
        self.assertIn("(1)", p1)
        self.assertIn("(2)", p1)
        self.assertIn("kekuatan peraturan pidana", p1.lower())
        self.assertIn("dilarang digunakan analogi", p1.lower())

        # Pasal 2: Ayat (1), (2), (3) with 'l2l' fixed to '(2)'
        p2 = self.articles[2]["isi_teks"]
        self.assertIn("(1)", p2)
        self.assertIn("(2)", p2)
        self.assertIn("(3)", p2)
        self.assertNotIn("l2l", p2, "OCR artifact 'l2l' was not converted to '(2)'")

        # Pasal 459: Pembunuhan Berencana
        p459 = self.articles[459]["isi_teks"]
        self.assertIn("pembunuhan berencana", p459.lower())
        self.assertIn("pidana mati", p459.lower())

        # Pasal 460: Should contain cross-reference without truncation
        p460 = self.articles[460]["isi_teks"]
        self.assertIn("Pasal 459", p460)

        # Pasal 624: Final article
        p624 = self.articles[624]["isi_teks"]
        self.assertIn("3 (tiga) tahun", p624)
        self.assertIn("terhitung sejak tanggal diundangkan", p624)
        self.assertNotIn("Agar", p624, "Catchword 'Agar' leaked into Pasal 624")


if __name__ == "__main__":
    unittest.main()
