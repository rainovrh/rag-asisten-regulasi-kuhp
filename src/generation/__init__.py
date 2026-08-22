"""Generation package for LLM inference."""

from src.generation.llm import LLMEngine
from src.generation.prompts import get_legal_qa_prompt

__all__ = ["LLMEngine", "get_legal_qa_prompt"]
