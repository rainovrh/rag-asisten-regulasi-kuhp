"""LLM interface for text generation."""

import re
from typing import Optional

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class GenerationError(Exception):
    """Raised when LLM generation fails."""


class LLMEngine:
    """LLM inference engine using Ollama."""
    
    REFUSAL_PATTERNS = [
        r"saya tidak dapat menemukan pasal yang relevan",
        r"tidak dapat menemukan",
        r"tidak ditemukan",
        r"tidak ada informasi",
        r"konteks tidak memuat",
        r"tidak memiliki informasi",
    ]
    
    def __init__(
        self,
        model: str = settings.llm_model,
        temperature: float = settings.llm_temperature,
        max_tokens: int = settings.llm_max_tokens,
    ) -> None:
        """Initialize LLM engine.
        
        Args:
            model: Ollama model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Initializing LLM: {model}")
        self._llm = Ollama(model=model, temperature=temperature)
    
    def generate(
        self,
        prompt: str,
        stop_sequences: Optional[list[str]] = None,
    ) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input prompt.
            stop_sequences: Optional sequences to stop generation.
        
        Returns:
            Generated text.
        
        Raises:
            GenerationError: If generation fails.
        """
        try:
            response = self._llm.invoke(prompt, stop=stop_sequences)
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise GenerationError(f"LLM generation failed: {e}") from e
    
    def generate_with_template(
        self,
        template: PromptTemplate,
        context: str,
        query: str,
    ) -> str:
        """Generate using a prompt template.
        
        Args:
            template: PromptTemplate instance.
            context: Context string.
            query: User query.
        
        Returns:
            Generated text.
        """
        prompt = template.format(context=context, query=query)
        return self.generate(prompt)
    
    def is_refusal(self, text: str) -> bool:
        """Check if the generated text is a refusal.
        
        Args:
            text: Generated text.
        
        Returns:
            True if the text appears to be a refusal.
        """
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in self.REFUSAL_PATTERNS)
