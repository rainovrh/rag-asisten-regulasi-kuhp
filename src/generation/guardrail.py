"""Strict Legal Grounding and Citation Check Guardrail (Blok 4, Kotak 16).

Verifies citation existence and strictly catches hallucinated articles:
- Regex-based article citation extractor.
- Cross-validation against retrieved context metadata.
- Citation Accuracy (%) and Hallucinated Citation Rate (%).
- Guardrail warning annotation / fallback enforcement.
"""

from dataclasses import dataclass, field
import re
from typing import List, Optional, Set, Tuple

from src.retrieval.hybrid_retriever import RetrievedLegalChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CitationDetails:
    """Detailed mention of an article citation in generated text."""

    raw_mention: str
    pasal_number: int
    ayat_mention: Optional[str] = None
    is_valid: bool = True


@dataclass
class GuardrailValidationResult:
    """Comprehensive validation result of strict legal grounding (Kotak 16)."""

    is_grounded: bool
    citation_accuracy: float
    hallucinated_citation_rate: float
    total_citations: int
    valid_citations: List[str]
    hallucinated_citations: List[str]
    sanitized_response: str
    warning_message: Optional[str] = None


class StrictLegalGroundingGuardrail:
    """Guardrail to enforce 100% Citation Accuracy and eliminate hallucinated legal citations."""

    # Regex to capture Indonesian article mentions: "Pasal 252", "Pasal 1 ayat (1)", "pasal 459 huruf a"
    CITATION_PATTERN = re.compile(
        r"\b(?:Pasal|pasal)\s*(\d{1,3})\b(?:\s*(?:ayat\s*\(([0-9a-zA-Z]+)\)|ayat\s*([0-9a-zA-Z]+)|huruf\s*([a-zA-Z])))?",
        re.IGNORECASE,
    )

    def __init__(self, auto_append_warning: bool = True) -> None:
        """Initialize strict grounding guardrail.

        Args:
            auto_append_warning: Whether to automatically attach warning banner if hallucinations occur.
        """
        self.auto_append_warning = auto_append_warning

    def extract_citations(self, text: str) -> List[CitationDetails]:
        """Extract all article citations from generated response.

        Args:
            text: LLM generated answer text.

        Returns:
            List of CitationDetails objects.
        """
        citations: List[CitationDetails] = []
        if not text:
            return citations

        matches = self.CITATION_PATTERN.finditer(text)
        for m in matches:
            raw_mention = m.group(0).strip()
            num_str = m.group(1)
            pasal_num = int(num_str)

            ayat = m.group(2) or m.group(3) or m.group(4)
            citations.append(
                CitationDetails(
                    raw_mention=raw_mention,
                    pasal_number=pasal_num,
                    ayat_mention=ayat,
                )
            )

        return citations

    def validate_grounding(
        self,
        llm_response: str,
        retrieved_chunks: List[RetrievedLegalChunk],
    ) -> GuardrailValidationResult:
        """Perform strict cross-verification of LLM response citations against retrieved chunks.

        Target:
        - 100% Citation Accuracy (0 fictitious/hallucinated citations).
        - Any article mentioned must exist in retrieved chunks context.

        Args:
            llm_response: Generated text from LLM.
            retrieved_chunks: Chunks passed in context window (Kotak 13).

        Returns:
            GuardrailValidationResult with metrics and verification details.
        """
        # 1. Collect all valid pasal numbers from context
        valid_pasal_numbers: Set[int] = set()
        for chunk in retrieved_chunks:
            if chunk.pasal:
                nums = re.findall(r"\b(\d{1,3})\b", chunk.pasal)
                for n in nums:
                    valid_pasal_numbers.add(int(n))

            nomor = chunk.metadata.get("nomor_pasal")
            if nomor is not None and isinstance(nomor, int):
                valid_pasal_numbers.add(nomor)

        # 2. Extract citations from response
        extracted_citations = self.extract_citations(llm_response)

        valid_citations: List[str] = []
        hallucinated_citations: List[str] = []

        for cit in extracted_citations:
            if cit.pasal_number in valid_pasal_numbers:
                cit.is_valid = True
                if cit.raw_mention not in valid_citations:
                    valid_citations.append(cit.raw_mention)
            else:
                cit.is_valid = False
                if cit.raw_mention not in hallucinated_citations:
                    hallucinated_citations.append(cit.raw_mention)

        total_citations = len(valid_citations) + len(hallucinated_citations)

        # 3. Compute metrics
        if total_citations == 0:
            # If no citations mentioned, accuracy is 100% unless answer claims facts without citation
            citation_accuracy = 1.0
            hallucinated_rate = 0.0
            is_grounded = True
        else:
            citation_accuracy = len(valid_citations) / float(total_citations)
            hallucinated_rate = len(hallucinated_citations) / float(total_citations)
            is_grounded = len(hallucinated_citations) == 0

        # 4. Generate warning or sanitized response if hallucinated
        sanitized_response = llm_response
        warning_message: Optional[str] = None

        if not is_grounded:
            warning_message = (
                f"⚠️ [GUARDRAIL WARNING] Ditemukan sitasi tidak berdasar: "
                f"{', '.join(hallucinated_citations)}. Pasal tersebut tidak terdapat dalam dokumen konteks yang diambil."
            )
            logger.warning(f"[KOTAK 16] Strict Grounding Violation: {warning_message}")

            if self.auto_append_warning:
                sanitized_response = f"{llm_response}\n\n---\n{warning_message}"
        else:
            logger.info(f"[KOTAK 16] Strict Grounding Check PASSED: 100% Citation Accuracy ({len(valid_citations)} citations valid)")

        return GuardrailValidationResult(
            is_grounded=is_grounded,
            citation_accuracy=citation_accuracy,
            hallucinated_citation_rate=hallucinated_rate,
            total_citations=total_citations,
            valid_citations=valid_citations,
            hallucinated_citations=hallucinated_citations,
            sanitized_response=sanitized_response,
            warning_message=warning_message,
        )

