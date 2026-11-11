"""Generation package for LLM inference (Blok 4: Kotak 14 s.d. 17)."""

from src.generation.guardrail import GuardrailValidationResult, StrictLegalGroundingGuardrail
from src.generation.llm_builder import GroqLLMEngine, LLMEngine, MockLLMEngine
from src.generation.prompts import AugmentedPromptBuilder, get_legal_qa_prompt
from src.generation.response_formatter import LegalAnswerOutput, LegalResponseFormatter, LiteralArticleQuote

__all__ = [
    "AugmentedPromptBuilder",
    "GroqLLMEngine",
    "LLMEngine",
    "MockLLMEngine",
    "StrictLegalGroundingGuardrail",
    "GuardrailValidationResult",
    "LegalResponseFormatter",
    "LegalAnswerOutput",
    "LiteralArticleQuote",
    "get_legal_qa_prompt",
]
