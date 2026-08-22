"""Tests for preprocessing module."""

import pytest
from src.preprocessing.cleaner import DocumentCleaner
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.chunker import SemanticChunker, FixedSizeChunker


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


class TestSemanticChunker:
    """Tests for SemanticChunker."""
    
    def test_chunk_returns_list(self) -> None:
        """Test that chunking returns a list."""
        chunker = SemanticChunker(threshold=0.5)
        text = "Ini adalah kalimat pertama. Ini adalah kalimat kedua."
        chunks = chunker.chunk(text)
        assert isinstance(chunks, list)
    
    def test_chunk_empty_text(self) -> None:
        """Test chunking empty text."""
        chunker = SemanticChunker()
        chunks = chunker.chunk("")
        assert len(chunks) == 0
