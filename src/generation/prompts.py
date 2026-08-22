"""Prompt templates for legal QA."""

from langchain_core.prompts import PromptTemplate

from src.utils.logger import get_logger

logger = get_logger(__name__)


LEGAL_QA_TEMPLATE = """Anda adalah pakar hukum AI yang sangat ketat dan objektif. 
Jawablah pertanyaan berikut HANYA berdasarkan DOKUMEN KONTEKS di bawah ini.
DILARANG memberikan asumsi, pendapat pribadi, atau saran di luar isi dokumen konteks.
Jelaskan unsur pidana atau sanksinya secara ringkas, formal, dan sebutkan nomor Pasalnya secara jelas.

DOKUMEN KONTEKS:
{context}

PERTANYAAN PENGGUNA:
{query}

JAWABAN RELEVAN:
"""


def get_legal_qa_prompt() -> PromptTemplate:
    """Get the legal QA prompt template.
    
    Returns:
        Configured PromptTemplate instance.
    """
    return PromptTemplate(
        input_variables=["context", "query"],
        template=LEGAL_QA_TEMPLATE,
    )
