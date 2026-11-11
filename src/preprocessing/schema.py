"""Pydantic Schema for Legal Chunks and Metadata Enrichment in KUHP RAG System."""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class LegalChunkMetadata(BaseModel):
    """Metadata enrichment for KUHP regulatory chunks."""
    model_config = ConfigDict(extra="forbid")

    id_unik: str = Field(
        ...,
        description="Unique identifier, e.g. 'kuhp_pasal_1', 'kuhp_pasal_622_ayat_1', 'kuhp_konsiderans'",
    )
    buku: str = Field(
        ...,
        description="KUHP Book level (e.g., 'BUKU KESATU ATURAN UMUM')",
    )
    bab: str = Field(
        ...,
        description="KUHP Chapter level (e.g., 'BAB I RUANG LINGKUP...')",
    )
    bagian: Optional[str] = Field(
        None,
        description="KUHP Section/Part (e.g., 'Bagian Kesatu Menurut Waktu')",
    )
    paragraf: Optional[str] = Field(
        None,
        description="KUHP Paragraph (e.g., 'Paragraf 1 Umum')",
    )
    pasal: str = Field(
        ...,
        description="Article designation, e.g. 'Pasal 1' or 'Konsiderans'",
    )
    nomor_pasal: Optional[int] = Field(
        None,
        description="Integer article number 1..624, or None for Konsiderans",
    )
    ayat: List[str] = Field(
        default_factory=list,
        description="List of clauses in chunk, e.g. ['(1)', '(2)'] or ['(1)']",
    )
    has_penjelasan: bool = Field(
        False,
        description="True if substantive authentic statutory explanation is present",
    )
    penjelasan: Optional[str] = Field(
        None,
        description="Substantive authentic statutory explanation text (None if Cukup Jelas)",
    )
    char_count: int = Field(
        ...,
        ge=1,
        description="Character count of chunk text",
    )
    token_count: int = Field(
        ...,
        ge=1,
        description="Word/token count of chunk text",
    )


class LegalChunk(BaseModel):
    """Complete Legal Document Chunk ready for Indexing and LLM Context Injection."""
    model_config = ConfigDict(extra="forbid")

    id_unik: str = Field(..., description="Unique ID matching metadata.id_unik")
    isi_teks: str = Field(
        ...,
        description="Formatted Tri-Block text ([KONTEKS REGULASI], [BUNYI PASAL ASLI], and [PENJELASAN AUTENTIK])",
    )
    metadata: LegalChunkMetadata = Field(..., description="Structured metadata attributes")

