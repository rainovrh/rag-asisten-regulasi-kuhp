"""LLM interface for text generation using Groq."""

import re
import time
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class GenerationError(Exception):
    """Raised when LLM generation fails."""


class LLMEngine:
    """LLM inference engine using Groq API."""
    
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
            model: Groq model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Initializing Groq LLM: {model}")
        
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables.")
            
        self._llm = ChatGroq(
            model_name=model, 
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=settings.groq_api_key
        )
    
    def generate(
        self,
        prompt: str,
        stop_sequences: Optional[list[str]] = None,
        max_retries: int = 3,
    ) -> str:
        """Generate text from prompt with retry on rate limit.
        
        Args:
            prompt: Input prompt.
            stop_sequences: Optional sequences to stop generation.
            max_retries: Number of retries on rate limit (429).
        
        Returns:
            Generated text.
        
        Raises:
            GenerationError: If generation fails after retries.
        """
        for attempt in range(max_retries):
            try:
                response = self._llm.invoke(prompt, stop=stop_sequences)
                return response.content
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"Rate limit hit. Retrying in {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM generation failed: {e}")
                    raise GenerationError(f"LLM generation failed: {e}") from e
        
        raise GenerationError("Max retries exceeded")
    
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
