"""Tests for generation module."""

import pytest
from src.generation.llm import LLMEngine
from src.generation.prompts import get_legal_qa_prompt


class TestLLMEngine:
    """Tests for LLMEngine."""
    
    def test_init_default_values(self) -> None:
        """Test initialization with defaults."""
        # Note: This test requires Ollama to be running
        # For CI, use mocking instead
        pass
    
    def test_generate_with_template(self) -> None:
        """Test template-based generation."""
        prompt = get_legal_qa_prompt()
        assert "DOKUMEN KONTEKS" in prompt.template
        assert "PERTANYAAN PENGGUNA" in prompt.template


class TestPrompts:
    """Tests for prompt templates."""
    
    def test_legal_qa_prompt_contains_required_elements(self) -> None:
        """Test that legal QA prompt has required elements."""
        prompt = get_legal_qa_prompt()
        template = prompt.template
        
        assert "DOKUMEN KONTEKS" in template
        assert "PERTANYAAN PENGGUNA" in template
        assert "{context}" in template
        assert "{query}" in template
