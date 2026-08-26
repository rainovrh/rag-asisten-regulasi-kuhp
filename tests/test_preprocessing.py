"""Tests for preprocessing module."""

import pytest
from src.preprocessing.cleaner import DocumentCleaner
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.chunker import SemanticChunker, FixedSizeChunker, PasalSegmenter


class TestDocumentCleaner:
    """Tests for DocumentCleaner."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.cleaner = DocumentCleaner()

    def test_clean_removes_headers(self) -> None:
        """Test that headers are removed."""
        text = "PRESIDEN REPUBLIK INDONESIA\nSome content\nPRESIDEN REPUBLIK INDONESIA"
        cleaned = self.cleaner.clean(text)
        assert "PRESIDEN" not in cleaned

    def test_clean_normalizes_whitespace(self) -> None:
        """Test whitespace normalization."""
        text = "Line 1\n\n\n\nLine 2"
        cleaned = self.cleaner.clean(text)
        assert "\n\n\n" not in cleaned

    def test_clean_removes_dash_style_page_numbers(self) -> None:
        """Test removal of '- N -' page number footers."""
        text = "akhir kalimat.\n- 12 -\nPasal 2 berikutnya"
        cleaned = self.cleaner.clean(text)
        assert "- 12 -" not in cleaned

    def test_clean_removes_ocr_mangled_sk_stamps(self) -> None:
        """Test SK footer stamps with OCR-mangled digits are removed safely."""
        text = "dibatasi oleh hal yang dikecualikan\nSK No l61015 A\nParagraf 2 Pengecualian"
        cleaned = self.cleaner.clean(text)
        assert "SK No" not in cleaned
        assert "Paragraf 2 Pengecualian" in cleaned

    def test_clean_removes_halaman_style_page_numbers(self) -> None:
        """Test removal of 'Halaman N' markers as specified in proposal."""
        text = "melakukan penganiayaan\nHalaman 12\ndiancam dengan pidana"
        cleaned = self.cleaner.clean(text)
        assert "Halaman 12" not in cleaned
        assert "penganiayaan" in cleaned and "diancam" in cleaned

    def test_clean_joins_sentence_broken_by_newline(self) -> None:
        """Test rejoining of mid-sentence PDF line breaks."""
        text = "hukum pidana nasional\nNegara Kesatuan Republik Indonesia"
        cleaned = self.cleaner.clean(text)
        assert "nasional Negara Kesatuan" in cleaned
        assert "\n" not in cleaned.split("nasional")[1].split("Kesatuan")[0]

    def test_clean_joins_multiple_broken_lines(self) -> None:
        """Test rejoining across several consecutive broken lines."""
        text = "perbuatan yang\ndapat dipidana\nmenurut undang-undang."
        cleaned = self.cleaner.clean(text)
        assert "perbuatan yang dapat dipidana menurut undang-undang." in cleaned

    def test_clean_preserves_newline_before_pasal(self) -> None:
        """Test structural boundary: newline before 'Pasal' is kept."""
        text = "diancam dengan pidana penjara\nPasal 2\nketentuan lebih lanjut"
        cleaned = self.cleaner.clean(text)
        assert "\nPasal 2" in cleaned

    def test_clean_preserves_newline_before_numbered_item(self) -> None:
        """Test structural boundary: numbered/lettered items keep their line."""
        text = "sebagai berikut:\n1. Ketentuan pertama\na. Ketentuan kedua"
        cleaned = self.cleaner.clean(text)
        assert "\n1. Ketentuan" in cleaned
        assert "\na. Ketentuan" in cleaned

    def test_clean_preserves_newline_after_sentence_end(self) -> None:
        """Test sentence-final period keeps the line separation."""
        text = "Ini kalimat pertama.\nIni kalimat kedua."
        cleaned = self.cleaner.clean(text)
        assert "pertama.\n" in cleaned

    def test_clean_dehyphenates_broken_compound_word(self) -> None:
        """Test hyphenated word split across lines is rejoined without gap."""
        text = "masyarakat bangsa-\nbangsa, perlu disusun"
        cleaned = self.cleaner.clean(text)
        assert "bangsa-bangsa," in cleaned
        assert "bangsa- bangsa" not in cleaned


class TestTextNormalizer:
    """Tests for TextNormalizer."""

    def test_normalize_lowercase(self) -> None:
        """Test lowercase conversion."""
        normalizer = TextNormalizer(lowercase=True)
        result = normalizer.normalize("HELLO WORLD")
        assert result == "hello world"

    def test_normalize_abbreviations(self) -> None:
        """Test abbreviation expansion."""
        normalizer = TextNormalizer(expand_abbreviations=True)
        result = normalizer.normalize("UU No 1 Tahun 2023")
        assert "undang-undang" in result

    def test_normalize_kuhp_expansion(self) -> None:
        """Test KUHP expansion preserves pasal numbering context."""
        normalizer = TextNormalizer(expand_abbreviations=True)
        result = normalizer.normalize("Pasal 1 KUHP berlaku")
        assert "kitab undang-undang hukum pidana" in result
        assert "Pasal 1" not in result or result.startswith("pasal 1")

    def test_normalize_does_not_expand_invalid_abbreviations(self) -> None:
        """Test removed/wrong mappings no longer corrupt words."""
        normalizer = TextNormalizer(expand_abbreviations=True)
        assert normalizer.normalize("KUP") == "kup"
        assert normalizer.normalize("pidana") == "pidana"

    def test_normalize_preserves_numbers_and_punctuation(self) -> None:
        """Test numbers and punctuation survive normalization."""
        normalizer = TextNormalizer()
        result = normalizer.normalize("UU No. 11 Tahun 2008.")
        assert "no. 11 tahun 2008." in result


class TestPasalSegmenter:
    """Tests for PasalSegmenter."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.segmenter = PasalSegmenter(line_anchored=True)

    def test_line_anchored_ignores_inline_cross_references(self) -> None:
        """Test inline 'dalam Pasal 4' references do not create segments."""
        text = (
            "Pasal 4\nPenerapan ketentuan dalam Pasal 2 dan Pasal 3\n"
            "Pasal 5\nKetentuan lain."
        )
        segments = self.segmenter.segment(text)
        ids = [s for s, _ in segments]
        assert ids == ["pasal 4", "pasal 5"]

    def test_segment_deduplicates_repeated_headings(self) -> None:
        """Test duplicate headings yield unique segment ids only once."""
        text = "Pasal 1\nIsi pertama.\nPasal 1\nIsi kedua."
        segments = self.segmenter.segment(text)
        assert len(segments) == 1

    def test_unanchored_mode_keeps_legacy_behavior(self) -> None:
        """Test default mode still matches inline references (legacy)."""
        segmenter = PasalSegmenter()
        segments = segmenter.segment("rujukan dalam Pasal 2 tetap cocok")
        assert len(segments) == 1


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.chunker = SemanticChunker(threshold=0.5)

    def test_chunk_single_segment_returns_one_chunk(self) -> None:
        """Test short segment yields a single chunk without model loading."""
        chunks = self.chunker.chunk_segments([("pasal 1", "Satu kalimat saja.")])
        assert isinstance(chunks, list)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "pasal 1_chunk_0"

    def test_chunk_empty_segment(self) -> None:
        """Test empty segment yields a placeholder chunk."""
        chunks = self.chunker.chunk_segments([("pasal 1", "")])
        assert len(chunks) == 1

    def test_split_sentences_on_punctuation(self) -> None:
        """Test sentence splitting on terminal punctuation."""
        sentences = self.chunker._split_sentences("Kalimat pertama. Kalimat kedua!")
        assert sentences == ["Kalimat pertama.", "Kalimat kedua!"]
