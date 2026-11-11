"""Augmented Prompt Builder for Legal QA (Blok 4, Kotak 14).

Constructs strictly grounded prompt windows leveraging Tri-Block chunks:
- [KONTEKS REGULASI]
- [BUNYI PASAL ASLI]
- [PENJELASAN AUTENTIK PEMBENTUK UU]
"""

from typing import Any, List, Optional
import re

from src.retrieval.hybrid_retriever import RetrievedLegalChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT_TEMPLATE = """Anda adalah Asisten Hukum AI Spesialis KUHP Baru (Undang-Undang Nomor 1 Tahun 2023).
Tugas Anda adalah memberikan jawaban hukum yang presisi, objektif, dan berwibawa berdasarkan HANYA pada naskah regulasi yang diberikan dalam DOKUMEN KONTEKS.

=== PRINSIP STRICT GROUNDING & ANTI-HALUSINASI ===
1. ZERO HALLUCINATION ON CITATIONS: Anda HANYA diperbolehkan menyebutkan nomor pasal dan ayat yang secara eksplisit tertulis di dalam DOKUMEN KONTEKS di bawah. DILARANG KERAS mengarang, mereka-reka nomor pasal lain, atau merujuk KUHP lama (W.v.S.).
2. PENALARAN BERDASARKAN DOKUMEN: Gunakan bagian [BUNYI PASAL ASLI] sebagai norma hukum yang mengikat dan bagian [PENJELASAN AUTENTIK PEMBENTUK UU] sebagai doktrin/tafsir resmi pembentuk undang-undang.
3. KETIDAKTERSEDIAAN INFORMASI: Jika pertanyaan pengguna tidak diatur atau tidak dapat dijawab oleh DOKUMEN KONTEKS yang disediakan, Anda WAJIB menjawab:
   "Ketentuan mengenai hal tersebut tidak diatur dalam KUHP Baru (UU No. 1 Tahun 2023) berdasarkan dokumen konteks yang tersedia."

=== STRUKTUR FORMAT JAWABAN ===
Format jawaban Anda WAJIB tersusun rapi menjadi 3 (tiga) bagian berikut:

### KESIMPULAN
[Tuliskan 1-2 kalimat ringkas dan to-the-point yang langsung menjawab inti pertanyaan pengguna: apakah boleh/tidak boleh, dipidana/tidak dipidana, beserta ancaman sanksi utamanya].

### DASAR HUKUM & PENALARAN
[Uraikan analisis penalaran hukum sistematis: unsur-unsur tindak pidana, kualifikasi delik, pengecualian, dan elaborasi tafsir resmi dari Penjelasan Autentik].

### KUTIPAN PASAL ASLI
[Tuliskan kutipan bunyi harfiah pasal asli yang menjadi dasar hukum keputusan di atas].
"""


USER_PROMPT_TEMPLATE = """=== DOKUMEN KONTEKS HUKUM (KUHP BARU UU NO. 1 TAHUN 2023) ===
{context}

=== PERTANYAAN PENGGUNA ===
{query}

JAWABAN HUKUM (Patuhi struktur KESIMPULAN, DASAR HUKUM & PENALARAN, dan KUTIPAN PASAL ASLI):
"""


class AugmentedPromptBuilder:
    """Builder for constructing strictly grounded legal prompts (Kotak 14)."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT_TEMPLATE) -> None:
        """Initialize prompt builder."""
        self.system_prompt = system_prompt

    def format_context(self, chunks: List[RetrievedLegalChunk]) -> str:
        """Format list of retrieved Tri-Block legal chunks into structured context text.

        Args:
            chunks: List of retrieved chunks from hybrid retrieval (Kotak 13).

        Returns:
            Structured context string.
        """
        if not chunks:
            return "[TIDAK ADA DOKUMEN KONTEKS YANG RELEVAN]"

        formatted_docs: List[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            pasal_label = chunk.pasal or chunk.chunk_id
            header = f"--- [DOKUMEN KONTEKS {idx}: {pasal_label}] ---"
            body = chunk.text.strip()
            formatted_docs.append(f"{header}\n{body}\n")

        return "\n".join(formatted_docs)

    def build_prompt(self, query: str, chunks: List[RetrievedLegalChunk]) -> str:
        """Build complete single-string prompt (compatible with legacy prompt templates)."""
        context_str = self.format_context(chunks)
        user_part = USER_PROMPT_TEMPLATE.format(context=context_str, query=query.strip())
        return f"{self.system_prompt}\n\n{user_part}"

    def build_messages(self, query: str, chunks: List[RetrievedLegalChunk]) -> List[dict]:
        """Build chat messages list for chat completions API (Groq/OpenAI compatible).

        Args:
            query: User legal question.
            chunks: Retrieved chunks.

        Returns:
            List of message dicts: [{'role': 'system', ...}, {'role': 'user', ...}]
        """
        context_str = self.format_context(chunks)
        user_content = USER_PROMPT_TEMPLATE.format(context=context_str, query=query.strip())

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def extract_context_pasal_numbers(chunks: List[RetrievedLegalChunk]) -> set[int]:
        """Extract all article numbers present in the retrieved chunks context.

        Returns:
            Set of integer article numbers (e.g. {1, 2, 252, 459}).
        """
        pasal_numbers: set[int] = set()
        for chunk in chunks:
            # Check metadata
            nomor = chunk.metadata.get("nomor_pasal")
            if nomor is not None and isinstance(nomor, int):
                pasal_numbers.add(nomor)

            # Check chunk.pasal string
            if chunk.pasal:
                matches = re.findall(r"\bpasal\s*(\d+)\b", chunk.pasal, re.IGNORECASE)
                for m in matches:
                    pasal_numbers.add(int(m))

            # Also check text for article headers
            matches_text = re.findall(r"(?:###\s*\[KONTEKS REGULASI\].*?Ketentuan:\s*Pasal\s*(\d+))", chunk.text, re.IGNORECASE | re.DOTALL)
            for m in matches_text:
                pasal_numbers.add(int(m))

        return pasal_numbers


# Legacy compatibility functions
def get_legal_qa_prompt():
    """Legacy helper for backward compatibility."""
    from langchain_core.prompts import PromptTemplate
    return PromptTemplate(
        input_variables=["context", "query"],
        template=SYSTEM_PROMPT_TEMPLATE + "\n\n" + USER_PROMPT_TEMPLATE,
    )
