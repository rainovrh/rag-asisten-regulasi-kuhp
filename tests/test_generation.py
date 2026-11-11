"""Unit Test Suite for Blok 4: LLM Answer Generation (Kotak 14 s.d. 17).

Strictly validates:
- Kotak 14: Augmented Prompt Builder (Tri-Block context injection, strict anti-hallucination prompts).
- Kotak 15: LLM Engine (Modular Groq configuration, Qwen model parameterization, Mock engine).
- Kotak 16: Strict Legal Grounding & Citation Check (100% Citation Accuracy, Hallucination Catching).
- Kotak 17: Legal Response Formatter (Structured Kesimpulan, Dasar Hukum, Kutipan Harfiah Pasal Asli).
"""

import unittest

from src.generation.guardrail import StrictLegalGroundingGuardrail
from src.generation.llm_builder import GroqLLMEngine, MockLLMEngine
from src.generation.prompts import AugmentedPromptBuilder
from src.generation.response_formatter import LegalResponseFormatter
from src.retrieval.hybrid_retriever import RetrievedLegalChunk


class TestBlok4Generation(unittest.TestCase):
    """Rigorous test suite for Blok 4 components."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create representative mock retrieved legal chunks for testing."""
        cls.chunk_252 = RetrievedLegalChunk(
            chunk_id="kuhp_pasal_252",
            score=0.032,
            text=(
                "### [KONTEKS REGULASI]\n"
                "Buku: BUKU KEDUA TINDAK PIDANA\n"
                "Bab: BAB V TINDAK PIDANA TERHADAP KETERTIBAN UMUM\n"
                "Ketentuan: Pasal 252\n\n"
                "### [BUNYI PASAL ASLI]\n"
                "(1) Setiap Orang yang menyatakan dirinya mempunyai kekuatan gaib, "
                "memberitahukan, memberikan harapan, menawarkan, atau memberikan bantuan jasa "
                "kepada orang lain bahwa karena perbuatannya dapat menimbulkan penyakit, kematian, "
                "atau penderitaan mental atau fisik seseorang, dipidana dengan pidana penjara paling lama 1 tahun 6 bulan.\n\n"
                "### [PENJELASAN AUTENTIK PEMBENTUK UU]\n"
                "Ketentuan ini dimaksudkan untuk mencegah praktik penipuan dan main hakim sendiri."
            ),
            metadata={
                "id_unik": "kuhp_pasal_252",
                "pasal": "Pasal 252",
                "nomor_pasal": 252,
                "buku": "BUKU KEDUA TINDAK PIDANA",
                "bab": "BAB V TINDAK PIDANA TERHADAP KETERTIBAN UMUM",
                "penjelasan": "Ketentuan ini dimaksudkan untuk mencegah praktik penipuan dan main hakim sendiri.",
            },
            rank=1,
            pasal="Pasal 252",
            bab="BAB V TINDAK PIDANA TERHADAP KETERTIBAN UMUM",
            buku="BUKU KEDUA TINDAK PIDANA",
            penjelasan="Ketentuan ini dimaksudkan untuk mencegah praktik penipuan dan main hakim sendiri.",
        )

        cls.chunk_1 = RetrievedLegalChunk(
            chunk_id="kuhp_pasal_1",
            score=0.031,
            text=(
                "### [KONTEKS REGULASI]\n"
                "Buku: BUKU KESATU ATURAN UMUM\n"
                "Bab: BAB I RUANG LINGKUP BERLAKUNYA HUKUM PIDANA\n"
                "Ketentuan: Pasal 1\n\n"
                "### [BUNYI PASAL ASLI]\n"
                "(1) Tidak ada satu perbuatan pun yang dapat dikenai sanksi pidana dan/atau tindakan, "
                "kecuali atas kekuatan peraturan pidana dalam peraturan perundang-undangan yang telah ada sebelum perbuatan dilakukan."
            ),
            metadata={
                "id_unik": "kuhp_pasal_1",
                "pasal": "Pasal 1",
                "nomor_pasal": 1,
                "buku": "BUKU KESATU ATURAN UMUM",
                "bab": "BAB I RUANG LINGKUP BERLAKUNYA HUKUM PIDANA",
            },
            rank=2,
            pasal="Pasal 1",
            bab="BAB I RUANG LINGKUP BERLAKUNYA HUKUM PIDANA",
            buku="BUKU KESATU ATURAN UMUM",
        )

        cls.chunks = [cls.chunk_252, cls.chunk_1]

    def test_01_kotak14_augmented_prompt_builder(self) -> None:
        """Kotak 14: Test prompt construction, context formatting, and article number extraction."""
        builder = AugmentedPromptBuilder()
        query = "Apakah seseorang yang mengaku punya kekuatan santet bisa dipidana?"

        # 1. Test format_context includes Tri-Block headers
        context_str = builder.format_context(self.chunks)
        self.assertIn("DOKUMEN KONTEKS 1: Pasal 252", context_str)
        self.assertIn("### [KONTEKS REGULASI]", context_str)
        self.assertIn("### [BUNYI PASAL ASLI]", context_str)
        self.assertIn("### [PENJELASAN AUTENTIK PEMBENTUK UU]", context_str)

        # 2. Test chat messages list structure
        messages = builder.build_messages(query, self.chunks)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("ZERO HALLUCINATION ON CITATIONS", messages[0]["content"])
        self.assertIn(query, messages[1]["content"])

        # 3. Test extraction of valid article numbers from chunks
        valid_nums = builder.extract_context_pasal_numbers(self.chunks)
        self.assertEqual(valid_nums, {1, 252})

    def test_02_kotak15_llm_engine_configuration(self) -> None:
        """Kotak 15: Test Groq LLM engine configuration parameters and Mock engine execution."""
        # 1. Verify GroqLLMEngine parameters
        engine = GroqLLMEngine(model="qwen/qwen3.8-27b", temperature=0.1, max_tokens=1024)
        self.assertEqual(engine.model, "qwen/qwen3.8-27b")
        self.assertEqual(engine.temperature, 0.1)
        self.assertEqual(engine.max_tokens, 1024)

        # 2. Test Mock engine generation (sandboxed offline safety)
        mock_engine = MockLLMEngine()
        response = mock_engine.generate("Tes pertanyaan hukum")
        self.assertIn("KESIMPULAN", response)
        self.assertIn("Pasal 252", response)

    def test_03_kotak16_strict_grounding_valid_citations(self) -> None:
        """Kotak 16: Test that valid citations produce 100% Citation Accuracy and is_grounded=True."""
        guardrail = StrictLegalGroundingGuardrail()

        # Valid text mentioning only Pasal 252 and Pasal 1
        valid_llm_text = (
            "Berdasarkan Pasal 252 ayat (1) KUHP Baru, seseorang yang menyatakan memiliki kekuatan gaib "
            "dapat dipidana. Hal ini sejalan dengan asas legalitas pada Pasal 1 ayat (1) KUHP."
        )

        res = guardrail.validate_grounding(valid_llm_text, self.chunks)
        self.assertTrue(res.is_grounded)
        self.assertEqual(res.citation_accuracy, 1.0)
        self.assertEqual(res.hallucinated_citation_rate, 0.0)
        self.assertIn("Pasal 252", res.valid_citations[0])
        self.assertEqual(len(res.hallucinated_citations), 0)
        self.assertIsNone(res.warning_message)

    def test_04_kotak16_strict_grounding_hallucination_detection(self) -> None:
        """Kotak 16: Test that hallucinated citations (not in context) are flagged and warning appended."""
        guardrail = StrictLegalGroundingGuardrail(auto_append_warning=True)

        # Hallucinated text mentioning Pasal 999 (not in context!)
        hallucinated_text = (
            "Pelaku santet diatur dalam Pasal 252 KUHP Baru. Selain itu, sanksi ganti rugi "
            "diatur dalam Pasal 999 ayat (2) KUHP."
        )

        res = guardrail.validate_grounding(hallucinated_text, self.chunks)
        self.assertFalse(res.is_grounded)
        # 1 valid citation (Pasal 252) + 1 hallucinated citation (Pasal 999) = 50% accuracy
        self.assertEqual(res.citation_accuracy, 0.5)
        self.assertEqual(res.hallucinated_citation_rate, 0.5)
        self.assertTrue(any("999" in h for h in res.hallucinated_citations))
        self.assertIsNotNone(res.warning_message)
        self.assertIn("⚠️ [GUARDRAIL WARNING]", res.sanitized_response)
        self.assertIn("999", res.sanitized_response)

    def test_05_kotak17_legal_response_formatter_and_literal_quotes(self) -> None:
        """Kotak 17: Test structured output formatting and verbatim literal article quotation."""
        guardrail = StrictLegalGroundingGuardrail()
        sample_query = "Apakah pelaku santet dapat dipidana?"

        sample_llm_text = (
            "### KESIMPULAN\n"
            "Ya, pelaku yang menyatakan memiliki kekuatan gaib untuk mencelakakan orang lain dapat dipidana penjara paling lama 1 tahun 6 bulan sesuai Pasal 252 KUHP Baru.\n\n"
            "### DASAR HUKUM & PENALARAN\n"
            "Ketentuan ini diatur dalam Buku Kedua Bab V Pasal 252 KUHP Baru untuk mencegah keresahan sosial.\n\n"
            "### KUTIPAN PASAL ASLI\n"
            "Pasal 252 ayat (1) KUHP Baru."
        )

        guardrail_res = guardrail.validate_grounding(sample_llm_text, self.chunks)
        output = LegalResponseFormatter.format(
            query=sample_query,
            llm_text=sample_llm_text,
            guardrail_result=guardrail_res,
            retrieved_chunks=self.chunks,
            inference_time=0.45,
        )

        # 1. Assert structured fields
        self.assertIn("1 tahun 6 bulan", output.kesimpulan)
        self.assertIn("Buku Kedua Bab V", output.dasar_hukum)
        self.assertEqual(output.grounding_status, "STRICTLY_GROUNDED")
        self.assertEqual(output.citation_accuracy, 1.0)
        self.assertTrue(output.is_strictly_grounded)

        # 2. Assert literal original quotes extraction
        self.assertEqual(len(output.kutipan_pasal), 2)
        quote_252 = next(q for q in output.kutipan_pasal if "252" in q.pasal)
        self.assertIn("Setiap Orang yang menyatakan dirinya mempunyai kekuatan gaib", quote_252.bunyi_pasal)
        self.assertEqual(quote_252.penjelasan, "Ketentuan ini dimaksudkan untuk mencegah praktik penipuan dan main hakim sendiri.")

        # 3. Assert dictionary serialization for API
        out_dict = output.to_dict()
        self.assertEqual(out_dict["query"], sample_query)
        self.assertEqual(len(out_dict["kutipan_pasal"]), 2)
        self.assertIn("full_markdown", out_dict)

    def test_06_end_to_end_blok4_pipeline(self) -> None:
        """End-to-End: Test complete flow Kotak 14 -> 15 -> 16 -> 17 using Mock engine."""
        engine = MockLLMEngine()
        query = "Sanksi untuk orang yang mengaku sakti atau dukun santet"

        output = engine.generate_legal_answer(query, self.chunks)

        self.assertIsNotNone(output.kesimpulan)
        self.assertIsNotNone(output.dasar_hukum)
        self.assertTrue(len(output.kutipan_pasal) > 0)
        self.assertTrue(output.is_strictly_grounded)
        self.assertEqual(output.citation_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
