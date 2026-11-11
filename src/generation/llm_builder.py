"""LLM Engine Integration supporting Groq API with Qwen & Llama models (Blok 4, Kotak 15).

Features:
- Groq Cloud API integration (tested on qwen/qwen3.8-27b).
- Modular model and API key configuration.
- Low temperature (0.1) for strictly minimized hallucinations.
- Rate-limit (HTTP 429) backoff retries.
- Offline/Mock engine support for sandboxed automated unit testing.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

from groq import Groq

from src.config.settings import settings
from src.generation.guardrail import StrictLegalGroundingGuardrail
from src.generation.prompts import AugmentedPromptBuilder
from src.generation.response_formatter import LegalAnswerOutput, LegalResponseFormatter
from src.retrieval.hybrid_retriever import RetrievedLegalChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GroqLLMEngine:
    """Production LLM Engine powered by Groq Cloud API (Kotak 15)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        """Initialize Groq LLM Engine.

        Args:
            model: Model identifier on Groq (default: settings.llm_model e.g. 'qwen/qwen3.8-27b').
            api_key: Groq API key (defaults to GROQ_API_KEY environment variable).
            temperature: Sampling temperature (default: 0.1 for strict factual grounding).
            max_tokens: Maximum tokens in generated completion (default: 1024).
        """
        self.model = model or settings.llm_model or "qwen/qwen3.8-27b"
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or settings.groq_api_key
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens or 1024

        if not self.api_key:
            logger.warning("GROQ_API_KEY is not set. GroqLLMEngine will fail on network calls unless mocked.")

        self._client: Optional[Groq] = None
        if self.api_key:
            self._client = Groq(api_key=self.api_key)

        # Child modules for Kotak 14, 16, 17
        self.prompt_builder = AugmentedPromptBuilder()
        self.guardrail = StrictLegalGroundingGuardrail(auto_append_warning=True)
        self.formatter = LegalResponseFormatter()

    def generate(
        self,
        prompt: str,
        max_retries: int = 3,
    ) -> str:
        """Generate text from single prompt string."""
        messages = [{"role": "user", "content": prompt}]
        return self.generate_chat(messages, max_retries=max_retries)

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 8,
    ) -> str:
        """Generate text from chat messages list with retry handling.

        Args:
            messages: List of message dictionaries [{'role': ..., 'content': ...}].
            max_retries: Number of retry attempts on transient errors.

        Returns:
            Generated response content.
        """
        if not self._client:
            raise RuntimeError("Groq client not initialized: GROQ_API_KEY is missing.")

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content or ""
                return content.strip()
            except Exception as e:
                err_msg = str(e)
                is_transient = any(
                    code in err_msg.lower()
                    for code in ["429", "503", "over capacity", "rate_limit", "rate limit", "timeout", "connection error"]
                )
                if is_transient and attempt < max_retries - 1:
                    # Dynamically extract wait time if provided by Groq (e.g. "try again in 14m55s" or "try again in 12s")
                    m = re.search(r"try again in (?:(\d+)m)?([\d\.]+)s", err_msg)
                    if m:
                        mins = float(m.group(1)) if m.group(1) else 0.0
                        secs = float(m.group(2)) if m.group(2) else 5.0
                        sleep_time = int(mins * 60 + secs) + 3
                    else:
                        sleep_time = 10 * (attempt + 1)
                    logger.warning(f"Groq API transient rate limit. Sleeping {sleep_time}s before retry (attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Groq API completion failed: {err_msg}")
                    raise

        raise RuntimeError("Failed to generate response after maximum retries.")

    def generate_legal_answer(
        self,
        query: str,
        retrieved_chunks: List[RetrievedLegalChunk],
    ) -> LegalAnswerOutput:
        """Execute full Blok 4 generation pipeline (Kotak 14 s.d. 17).

        Pipeline:
        1. Kotak 14: Augmented Prompt Builder (Tri-Block context injection)
        2. Kotak 15: LLM Groq API inference
        3. Kotak 16: Strict Legal Grounding & Citation Check Guardrail
        4. Kotak 17: Legal Response Formatter (Structured JSON & Markdown)

        Args:
            query: User inquiry.
            retrieved_chunks: Chunks from Kotak 13.

        Returns:
            LegalAnswerOutput object.
        """
        # Kotak 14: Build messages
        messages = self.prompt_builder.build_messages(query, retrieved_chunks)

        # Kotak 15: Infer LLM
        t0 = time.perf_counter()
        raw_llm_text = self.generate_chat(messages)
        inference_time = time.perf_counter() - t0

        # Kotak 16: Strict Grounding Guardrail
        guardrail_result = self.guardrail.validate_grounding(raw_llm_text, retrieved_chunks)

        # Kotak 17: Structured Response Formatting
        return self.formatter.format(
            query=query,
            llm_text=raw_llm_text,
            guardrail_result=guardrail_result,
            retrieved_chunks=retrieved_chunks,
            inference_time=inference_time,
        )


class MockLLMEngine(GroqLLMEngine):
    """Mock LLM Engine for offline unit tests in standard sandboxed environment."""

    def __init__(self, predefined_response: Optional[str] = None) -> None:
        """Initialize mock engine."""
        super().__init__(model="mock-qwen-27b", api_key="mock_key")
        self.predefined_response = predefined_response

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
    ) -> str:
        """Return predefined or context-grounded simulated response without network calls."""
        if self.predefined_response:
            return self.predefined_response

        # Default synthetic compliant response
        return (
            "### KESIMPULAN\n"
            "Seseorang yang menyatakan dirinya memiliki kekuatan gaib untuk menimbulkan penyakit dipidana penjara paling lama 1 (satu) tahun 6 (enam) bulan atau denda kategori IV berdasarkan Pasal 252 ayat (1) KUHP Baru.\n\n"
            "### DASAR HUKUM & PENALARAN\n"
            "Tindak pidana ini diatur dalam Buku Kedua Bab V Pasal 252 KUHP Baru. Ketentuan ini melarang perbuatan menawarkan jasa atau menyatakan diri memiliki kemampuan gaib yang dapat mencelakakan orang lain guna mencegah praktik penipuan dan main hakim sendiri.\n\n"
            "### KUTIPAN PASAL ASLI\n"
            "Pasal 252 ayat (1): Setiap Orang yang menyatakan dirinya mempunyai kekuatan gaib..."
        )


# Backward-compatible alias
LLMEngine = GroqLLMEngine

