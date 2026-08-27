"""LLM interface for text generation supporting Groq and Ollama."""

import re
import time
from typing import Optional

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from langchain_core.prompts import PromptTemplate

from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class GenerationError(Exception):
    """Raised when LLM generation fails."""


class LLMEngine:
    """LLM inference engine supporting Groq API and Ollama local models."""
    
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
        provider: str = "groq"
    ) -> None:
        """Initialize LLM engine.
        
        Args:
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            provider: LLM provider ("groq" or "ollama").
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider.lower()
        
        if self.provider == "groq":
            logger.info(f"Initializing Groq LLM: {model}")
            if not GROQ_AVAILABLE:
                raise ImportError("langchain-groq is not installed.")
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is not set in environment variables.")
            self._llm = ChatGroq(
                model_name=model, 
                temperature=temperature,
                max_tokens=max_tokens,
                groq_api_key=settings.groq_api_key
            )
        else:
            logger.info(f"Initializing local Ollama LLM: {model}")
            if not OLLAMA_AVAILABLE:
                raise ImportError("langchain-community is not installed.")
            self._llm = Ollama(model=model, temperature=temperature)
    
    def generate(
        self,
        prompt: str,
        stop_sequences: Optional[list[str]] = None,
        max_retries: int = 3,
    ) -> str:
        """Generate text from prompt."""
        for attempt in range(max_retries):
            try:
                if self.provider == "groq":
                    response = self._llm.invoke(prompt, stop=stop_sequences)
                    return response.content
                else:
                    return self._llm.invoke(prompt, stop=stop_sequences)
            except Exception as e:
                error_str = str(e)
                if self.provider == "groq" and "429" in error_str and attempt < max_retries - 1:
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
