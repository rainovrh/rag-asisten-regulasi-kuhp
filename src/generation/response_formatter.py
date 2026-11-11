"""Structured Legal Response Formatter (Blok 4, Kotak 17).

Formats final legal answer combining:
- Kesimpulan (Concise executive conclusion)
- Dasar Hukum & Penalaran (Legal reasoning)
- Kutipan Pasal Asli (Literal verbatim article text & official explanation)
- Strict Grounding & Citation Validation Status
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

from src.generation.guardrail import GuardrailValidationResult
from src.retrieval.hybrid_retriever import RetrievedLegalChunk


@dataclass
class LiteralArticleQuote:
    """Literal verbatim quotation of an original KUHP article."""

    pasal: str
    buku: str
    bab: str
    bunyi_pasal: str
    penjelasan: Optional[str] = None


@dataclass
class LegalAnswerOutput:
    """Final structured legal response (Kotak 17)."""

    query: str
    kesimpulan: str
    dasar_hukum: str
    kutipan_pasal: List[LiteralArticleQuote]
    sitasi_valid: List[str]
    citation_accuracy: float
    is_strictly_grounded: bool
    grounding_status: str
    full_markdown: str
    warning_message: Optional[str] = None
    inference_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for FastAPI endpoints and serialization."""
        return {
            "query": self.query,
            "kesimpulan": self.kesimpulan,
            "dasar_hukum": self.dasar_hukum,
            "kutipan_pasal": [
                {
                    "pasal": q.pasal,
                    "buku": q.buku,
                    "bab": q.bab,
                    "bunyi_pasal": q.bunyi_pasal,
                    "penjelasan": q.penjelasan,
                }
                for q in self.kutipan_pasal
            ],
            "sitasi_valid": self.sitasi_valid,
            "citation_accuracy": self.citation_accuracy,
            "is_strictly_grounded": self.is_strictly_grounded,
            "grounding_status": self.grounding_status,
            "warning_message": self.warning_message,
            "inference_time_seconds": self.inference_time_seconds,
            "full_markdown": self.full_markdown,
        }


class LegalResponseFormatter:
    """Formats raw LLM output and retrieved context into transparent structured legal answers (Kotak 17)."""

    @classmethod
    def format(
        cls,
        query: str,
        llm_text: str,
        guardrail_result: GuardrailValidationResult,
        retrieved_chunks: List[RetrievedLegalChunk],
        inference_time: float = 0.0,
    ) -> LegalAnswerOutput:
        """Parse sections from LLM response and extract original literal quotes from retrieved chunks.

        Args:
            query: User legal question.
            llm_text: Generated raw text from LLM.
            guardrail_result: Verification from Kotak 16.
            retrieved_chunks: Retrieved context chunks from Kotak 13.
            inference_time: LLM generation latency in seconds.

        Returns:
            LegalAnswerOutput object.
        """
        kesimpulan = ""
        dasar_hukum = ""
        kutipan_section = ""

        # Parse sections based on header patterns
        kesimpulan_match = re.search(
            r"###?\s*KESIMPULAN\s*\n(.*?)(?=\n###?\s*DASAR HUKUM|\n###?\s*KUTIPAN|\Z)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        dasar_hukum_match = re.search(
            r"###?\s*DASAR HUKUM\s*(?:&|DAN)?\s*PENALARAN\s*\n(.*?)(?=\n###?\s*KUTIPAN|\Z)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )
        kutipan_match = re.search(
            r"###?\s*KUTIPAN\s*PASAL\s*ASLI\s*\n(.*)",
            llm_text,
            re.IGNORECASE | re.DOTALL,
        )

        if kesimpulan_match:
            kesimpulan = kesimpulan_match.group(1).strip()
        if dasar_hukum_match:
            dasar_hukum = dasar_hukum_match.group(1).strip()
        if kutipan_match:
            kutipan_section = kutipan_match.group(1).strip()

        # Fallback if LLM generated freeform without exact headers
        if not kesimpulan and not dasar_hukum:
            kesimpulan = llm_text[:250].strip() + "..."
            dasar_hukum = llm_text.strip()

        # Extract literal quotes directly from verified retrieved chunks for 100% transparency
        literal_quotes: List[LiteralArticleQuote] = []
        seen_pasals = set()

        for chunk in retrieved_chunks:
            p_label = chunk.pasal or chunk.chunk_id
            if p_label in seen_pasals:
                continue
            seen_pasals.add(p_label)

            # Extract bunyi pasal asli from tri-block chunk text
            bunyi_match = re.search(r"###\s*\[BUNYI PASAL ASLI\]\s*\n(.*?)(?=\n###\s*\[PENJELASAN|\Z)", chunk.text, re.DOTALL)
            bunyi_text = bunyi_match.group(1).strip() if bunyi_match else chunk.text

            penjelasan_text = chunk.penjelasan

            literal_quotes.append(
                LiteralArticleQuote(
                    pasal=p_label,
                    buku=chunk.buku,
                    bab=chunk.bab,
                    bunyi_pasal=bunyi_text,
                    penjelasan=penjelasan_text,
                )
            )

        grounding_status = "STRICTLY_GROUNDED" if guardrail_result.is_grounded else "WARNING_UNGROUNDED_CITATIONS"

        # Build clean user-facing markdown
        md_lines = [
            "### ⚖️ KESIMPULAN HUKUM",
            kesimpulan,
            "",
            "### 📖 DASAR HUKUM & PENALARAN",
            dasar_hukum,
            "",
            "### 📜 BUNYI PASAL ASLI (RUJUKAN KUHP BARU UU NO. 1 TAHUN 2023)",
        ]

        for q in literal_quotes:
            md_lines.append(f"#### **{q.pasal}** ({q.buku} - {q.bab})")
            md_lines.append(f"> {q.bunyi_pasal}")
            if q.penjelasan and q.penjelasan != "Cukup jelas":
                md_lines.append(f"\n*Penjelasan Autentik Pembentuk UU:*  \n_{q.penjelasan}_")
            md_lines.append("")

        if not guardrail_result.is_grounded and guardrail_result.warning_message:
            md_lines.append("---")
            md_lines.append(guardrail_result.warning_message)

        full_markdown = "\n".join(md_lines)

        return LegalAnswerOutput(
            query=query,
            kesimpulan=kesimpulan,
            dasar_hukum=dasar_hukum,
            kutipan_pasal=literal_quotes,
            sitasi_valid=guardrail_result.valid_citations,
            citation_accuracy=guardrail_result.citation_accuracy,
            is_strictly_grounded=guardrail_result.is_grounded,
            grounding_status=grounding_status,
            full_markdown=full_markdown,
            warning_message=guardrail_result.warning_message,
            inference_time_seconds=inference_time,
        )

