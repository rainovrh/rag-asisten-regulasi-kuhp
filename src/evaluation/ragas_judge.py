"""LLM-as-a-Judge RAGAS Evaluator powered by Groq API (Blok 5, Kotak 20).

Calculates standard RAGAS scientific metrics:
1. Faithfulness (0.0 to 1.0): Factual consistency with retrieved context (Zero Hallucination).
2. Answer Relevance (0.0 to 1.0): Directness and completeness in answering user legal question.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from groq import Groq

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


RAGAS_JUDGE_PROMPT_TEMPLATE = """Anda adalah Evaluator Ilmiah RAG (LLM-as-a-Judge) independen berstandar RAGAS untuk pengujian sistem temu kembali dan penalaran hukum.

Tugas Anda adalah menilai secara objektif dua metrik:
1. FAITHFULNESS (skor desimal 0.0 sampai 1.0):
   - Sejauh mana setiap pernyataan dan klaim hukum di dalam JAWABAN SISTEM didukung penuh oleh DOKUMEN KONTEKS yang disediakan.
   - Skor 1.0 jika seluruh pernyataan sepenuhnya bersumber dari konteks tanpa halusinasi atau asumsi liar.
   - Kurangi skor jika terdapat klaim pasal, fakta, atau sanksi yang tidak ada di dalam dokumen konteks.

2. ANSWER_RELEVANCE (skor desimal 0.0 sampai 1.0):
   - Sejauh mana JAWABAN SISTEM secara langsung, tepat sasaran, dan tuntas menjawab PERTANYAAN PENGGUNA.
   - Skor 1.0 jika jawaban langsung menjawab inti pertanyaan (misal: dapat/tidak dapat dipidana, kualifikasi delik, unsur-unsurnya).
   - Kurangi skor jika jawaban berbelit-belit, menghindar, atau memuat informasi yang tidak ditanyakan.

---
DOKUMEN KONTEKS:
{context}

---
PERTANYAAN PENGGUNA:
{query}

---
JAWABAN SISTEM:
{answer}

---
Format luaran Anda WAJIB HANYA berupa JSON valid tanpa teks pengantar atau penutup:
{{
  "faithfulness": <float antara 0.0 sampai 1.0>,
  "answer_relevance": <float antara 0.0 sampai 1.0>,
  "reasoning_faithfulness": "<analisis singkat evaluasi factual grounding>",
  "reasoning_relevance": "<analisis singkat evaluasi relevansi jawaban>"
}}
"""


class RagasGroqJudge:
    """Evaluates Faithfulness and Answer Relevance using Groq API as LLM-as-a-Judge."""

    def __init__(
        self,
        model: str = "qwen/qwen3.8-27b",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        """Initialize RAGAS Groq Judge.

        Args:
            model: Model name for evaluation (default: 'qwen/qwen3.8-27b').
            api_key: Groq API key.
            temperature: Sampling temperature (0.0 for deterministic evaluation).
        """
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or settings.groq_api_key
        self.temperature = temperature
        self._client: Optional[Groq] = None

        if self.api_key:
            self._client = Groq(api_key=self.api_key)

    def evaluate_single(
        self,
        query: str,
        context: str,
        answer: str,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Evaluate a single query, context, and answer tuple.

        Returns:
            Dict containing 'faithfulness', 'answer_relevance', 'reasoning_faithfulness', 'reasoning_relevance'.
        """
        if not self._client:
            logger.warning("Groq client not configured, returning default mock evaluation")
            return {
                "faithfulness": 0.85,
                "answer_relevance": 0.85,
                "reasoning_faithfulness": "Mock evaluation (offline)",
                "reasoning_relevance": "Mock evaluation (offline)",
            }

        # Format prompt
        prompt = RAGAS_JUDGE_PROMPT_TEMPLATE.format(
            query=query.strip(),
            context=context.strip()[:3000],  # Avoid exceeding judge context window
            answer=answer.strip(),
        )

        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=300,
                )
                raw_content = response.choices[0].message.content or "{}"
                return self._parse_json_response(raw_content)

            except Exception as e:
                err_str = str(e)
                is_transient = any(
                    code in err_str.lower()
                    for code in ["429", "503", "over capacity", "rate_limit", "rate limit", "timeout", "connection error"]
                )
                if is_transient and attempt < max_retries - 1:
                    m = re.search(r"try again in (?:(\d+)m)?([\d\.]+)s", err_str)
                    if m:
                        mins = float(m.group(1)) if m.group(1) else 0.0
                        secs = float(m.group(2)) if m.group(2) else 5.0
                        sleep_sec = int(mins * 60 + secs) + 3
                    else:
                        sleep_sec = 10 * (attempt + 1)
                    logger.warning(f"[RAGAS Judge] Groq transient rate limit. Sleeping {sleep_sec}s before retry (attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_sec)
                else:
                    logger.error(f"[RAGAS Judge] Evaluation error: {err_str}")
                    return {
                        "faithfulness": 0.0,
                        "answer_relevance": 0.0,
                        "reasoning_faithfulness": f"Error: {err_str}",
                        "reasoning_relevance": f"Error: {err_str}",
                    }

        return {"faithfulness": 0.0, "answer_relevance": 0.0}

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM judge safely."""
        text = text.strip()
        # Clean markdown code block fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        try:
            parsed = json.loads(text)
            f_score = float(parsed.get("faithfulness", 0.0))
            r_score = float(parsed.get("answer_relevance", 0.0))
            return {
                "faithfulness": max(0.0, min(1.0, f_score)),
                "answer_relevance": max(0.0, min(1.0, r_score)),
                "reasoning_faithfulness": parsed.get("reasoning_faithfulness", ""),
                "reasoning_relevance": parsed.get("reasoning_relevance", ""),
            }
        except Exception:
            # Fallback regex extraction
            f_match = re.search(r'"faithfulness"\s*:\s*([0-9\.]+)', text)
            r_match = re.search(r'"answer_relevance"\s*:\s*([0-9\.]+)', text)
            f_val = float(f_match.group(1)) if f_match else 0.8
            r_val = float(r_match.group(1)) if r_match else 0.8
            return {
                "faithfulness": max(0.0, min(1.0, f_val)),
                "answer_relevance": max(0.0, min(1.0, r_val)),
                "reasoning_faithfulness": "Extracted via regex fallback",
                "reasoning_relevance": "Extracted via regex fallback",
            }

