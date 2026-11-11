"""Hierarchical Semantic Chunker and Metadata Enrichment Engine for KUHP.

Features:
1. Tri-Block Structured Formatting:
   - ### [KONTEKS REGULASI] (Hierarchical Breadcrumb)
   - ### [BUNYI PASAL ASLI] (Statutory Positive Law)
   - ### [PENJELASAN AUTENTIK PEMBENTUK UU] (Authentic Statutory Explanation)
2. Dynamic Slicing:
   - Articles <= 500 tokens: Kept intact at the 'Pasal' level.
   - Articles > 500 tokens (e.g. Pasal 622): Sliced at the 'Ayat' level without sentence breaking.
3. Metadata Enrichment:
   - Rigorously validates 100% of generated chunks against Pydantic LegalChunk schema.
"""

import json
from pathlib import Path
import re
from typing import Any, List, Optional, Tuple

from src.config.paths import PROCESSED_DATA_DIR
from src.preprocessing.schema import LegalChunk, LegalChunkMetadata
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HierarchicalChunker:
    """Chunks cleaned KUHP corpus maintaining complete hierarchical integrity."""

    # Heading detection patterns to separate structural titles trailing previous articles
    HEADING_PREFIX_PATTERN = re.compile(
        r'^(?:BUKU\s+[A-Z]+|BAB\s+[IVXLCDM]+|Bagian\s+[A-Za-z]+|Paragraf\s+\d+)',
        re.IGNORECASE,
    )

    def __init__(self, max_token_threshold: int = 500) -> None:
        """Initialize chunker.

        Args:
            max_token_threshold: Word/token threshold above which an article is
                                 dynamically partitioned into ayat-level chunks.
        """
        self.max_token_threshold = max_token_threshold

    @staticmethod
    def build_tri_block_text(
        buku: str,
        bab: str,
        pasal: str,
        body_text: str,
        bagian: Optional[str] = None,
        paragraf: Optional[str] = None,
        ayat_label: Optional[str] = None,
        penjelasan: Optional[str] = None,
    ) -> str:
        """Construct unambiguous Tri-Block formatted text for retrieval and LLM inference."""
        context_lines = [
            "### [KONTEKS REGULASI]",
            f"Buku: {buku}",
            f"Bab: {bab}",
        ]
        if bagian:
            context_lines.append(f"Bagian: {bagian}")
        if paragraf:
            context_lines.append(f"Paragraf: {paragraf}")

        target_title = f"{pasal} Ayat {ayat_label}" if ayat_label else pasal
        context_lines.append(f"Ketentuan: {target_title}")

        sections = [
            '\n'.join(context_lines),
            f"### [BUNYI PASAL ASLI]\n{body_text}",
        ]

        if penjelasan and penjelasan.strip():
            sections.append(f"### [PENJELASAN AUTENTIK PEMBENTUK UU]\n{penjelasan.strip()}")

        return '\n\n'.join(sections).strip()

    def separate_trailing_headings(self, raw_text: str) -> Tuple[str, List[str]]:
        """Separate structural headings (BUKU/BAB/Bagian/Paragraf) appearing at the tail of article text."""
        lines = raw_text.split('\n')
        heading_indices: List[int] = []

        # Scan backwards from bottom
        for j in range(len(lines) - 1, -1, -1):
            line = lines[j].strip()
            if self.HEADING_PREFIX_PATTERN.match(line):
                heading_indices.append(j)
            else:
                if heading_indices:
                    break

        if heading_indices:
            split_idx = min(heading_indices)
            clean_body = '\n'.join(lines[:split_idx]).strip()
            trailing_headings = [lines[k].strip() for k in range(split_idx, len(lines)) if lines[k].strip()]
            return clean_body, trailing_headings

        return raw_text.strip(), []

    def chunk_konsiderans(self, konsiderans_data: dict[str, Any]) -> LegalChunk:
        """Create structured chunk for Konsiderans (Menimbang & Mengingat)."""
        body_text = konsiderans_data["isi_teks"]
        isi_teks = (
            "### [KONTEKS REGULASI]\n"
            "Buku: PENDAHULUAN\n"
            "Bab: KONSIDERANS DAN ASAS FILOSOFIS\n"
            "Ketentuan: Konsiderans (Menimbang dan Mengingat)\n\n"
            f"### [BUNYI PASAL ASLI]\n{body_text}"
        )

        metadata = LegalChunkMetadata(
            id_unik="kuhp_konsiderans",
            buku="PENDAHULUAN",
            bab="KONSIDERANS DAN ASAS FILOSOFIS",
            bagian=None,
            paragraf=None,
            pasal="Konsiderans",
            nomor_pasal=None,
            ayat=[],
            has_penjelasan=False,
            penjelasan=None,
            char_count=len(isi_teks),
            token_count=len(isi_teks.split()),
        )

        return LegalChunk(
            id_unik="kuhp_konsiderans",
            isi_teks=isi_teks,
            metadata=metadata,
        )

    def chunk_articles(self, articles_dict: dict[Any, dict[str, Any]]) -> List[LegalChunk]:
        """Convert all articles into enriched LegalChunk objects."""
        # Initial hierarchy states for KUHP
        current_buku = "BUKU KESATU ATURAN UMUM"
        current_bab = "BAB I RUANG LINGKUP BERLAKUNYA KETENTUAN PERATURAN PERUNDANG-UNDANGAN PIDANA"
        current_bagian = "Bagian Kesatu Menurut Waktu"
        current_paragraf: Optional[str] = None

        chunks: List[LegalChunk] = []

        # Sort articles numerically by key
        sorted_keys = sorted(articles_dict.keys(), key=lambda x: int(x))

        for key in sorted_keys:
            num = int(key)
            art = articles_dict[key]
            raw_body = art["isi_teks"]
            penjelasan = art.get("penjelasan")

            # Clean body and extract trailing headings for the next article
            clean_body, next_headings = self.separate_trailing_headings(raw_body)
            token_count = len(clean_body.split())

            # Case A: Standard Article (<= max_token_threshold) -> Pasal Level Chunk
            if token_count <= self.max_token_threshold:
                ayat_matches = re.findall(r'(?m)^\((\d+)\)', clean_body)
                ayat_list = [f"({m})" for m in ayat_matches] if ayat_matches else ["(1)"]

                chunk_id = f"kuhp_pasal_{num}"
                formatted_text = self.build_tri_block_text(
                    buku=current_buku,
                    bab=current_bab,
                    bagian=current_bagian,
                    paragraf=current_paragraf,
                    pasal=f"Pasal {num}",
                    body_text=clean_body,
                    penjelasan=penjelasan,
                )

                metadata = LegalChunkMetadata(
                    id_unik=chunk_id,
                    buku=current_buku,
                    bab=current_bab,
                    bagian=current_bagian,
                    paragraf=current_paragraf,
                    pasal=f"Pasal {num}",
                    nomor_pasal=num,
                    ayat=ayat_list,
                    has_penjelasan=bool(penjelasan and penjelasan.strip()),
                    penjelasan=penjelasan,
                    char_count=len(formatted_text),
                    token_count=len(formatted_text.split()),
                )

                chunks.append(LegalChunk(id_unik=chunk_id, isi_teks=formatted_text, metadata=metadata))

            # Case B: Composite Long Article (> max_token_threshold) -> Dynamic Ayat-Level Slicing
            else:
                logger.info(
                    f"Article Pasal {num} exceeds threshold ({token_count} words). "
                    "Applying dynamic Ayat-level slicing."
                )
                ayat_splits = re.split(r'(?m)^(?=\(\d+\))', clean_body)
                for part in ayat_splits:
                    part_stripped = part.strip()
                    if not part_stripped:
                        continue

                    m_ayat = re.match(r'^\((\d+)\)', part_stripped)
                    ayat_label = f"({m_ayat.group(1)})" if m_ayat else None
                    ayat_num_clean = m_ayat.group(1) if m_ayat else "0"

                    chunk_id = f"kuhp_pasal_{num}_ayat_{ayat_num_clean}"
                    formatted_text = self.build_tri_block_text(
                        buku=current_buku,
                        bab=current_bab,
                        bagian=current_bagian,
                        paragraf=current_paragraf,
                        pasal=f"Pasal {num}",
                        ayat_label=ayat_label,
                        body_text=part_stripped,
                        penjelasan=penjelasan,
                    )

                    metadata = LegalChunkMetadata(
                        id_unik=chunk_id,
                        buku=current_buku,
                        bab=current_bab,
                        bagian=current_bagian,
                        paragraf=current_paragraf,
                        pasal=f"Pasal {num}",
                        nomor_pasal=num,
                        ayat=[ayat_label] if ayat_label else [],
                        has_penjelasan=bool(penjelasan and penjelasan.strip()),
                        penjelasan=penjelasan,
                        char_count=len(formatted_text),
                        token_count=len(formatted_text.split()),
                    )

                    chunks.append(LegalChunk(id_unik=chunk_id, isi_teks=formatted_text, metadata=metadata))

            # Update hierarchy for the subsequent articles from trailing headings
            for h in next_headings:
                h_clean = h.strip()
                if re.match(r'^BUKU\s+[A-Z]+', h_clean, re.I):
                    current_buku = h_clean
                    current_bab = "ATURAN UMUM" if "KESATU" in h_clean else "TINDAK PIDANA"
                    current_bagian = None
                    current_paragraf = None
                elif re.match(r'^BAB\s+[IVXLCDM]+', h_clean, re.I):
                    current_bab = h_clean
                    current_bagian = None
                    current_paragraf = None
                elif re.match(r'^Bagian\s+[A-Za-z]+', h_clean, re.I):
                    current_bagian = h_clean
                    current_paragraf = None
                elif re.match(r'^Paragraf\s+\d+', h_clean, re.I):
                    current_paragraf = h_clean

        return chunks

    def process(self, corpus_path: Optional[Path] = None) -> List[LegalChunk]:
        """Load corpus JSON and transform into full list of validated LegalChunks."""
        if corpus_path is None:
            corpus_path = PROCESSED_DATA_DIR / "kuhp_corpus_complete.json"

        if not corpus_path.exists():
            raise FileNotFoundError(f"Clean corpus not found at {corpus_path}")

        logger.info(f"Loading cleaned corpus from {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks: List[LegalChunk] = []

        # 1. Chunk Konsiderans
        if "konsiderans" in data:
            chunks.append(self.chunk_konsiderans(data["konsiderans"]))

        # 2. Chunk Articles (Pasal 1-624)
        if "articles" in data:
            article_chunks = self.chunk_articles(data["articles"])
            chunks.extend(article_chunks)

        logger.info(f"Hierarchical Chunking Complete: Generated {len(chunks)} enriched chunks")
        return chunks

    def save_chunks(
        self,
        chunks: List[LegalChunk],
        output_path: Optional[Path] = None,
        mapping_path: Optional[Path] = None,
    ) -> None:
        """Save chunks and pasal-to-chunk index mapping to disk."""
        if output_path is None:
            output_path = PROCESSED_DATA_DIR / "kuhp_chunks_hierarchical.json"
        if mapping_path is None:
            mapping_path = PROCESSED_DATA_DIR / "pasal_to_chunks.json"

        # Serialize Pydantic chunks to JSON
        serializable = [c.model_dump() for c in chunks]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

        # Build reverse index for retrieval evaluation (pasal -> list of chunk IDs)
        pasal_to_chunk_ids: dict[str, List[str]] = {}
        for c in chunks:
            p_key = c.metadata.pasal.lower()
            if p_key not in pasal_to_chunk_ids:
                pasal_to_chunk_ids[p_key] = []
            pasal_to_chunk_ids[p_key].append(c.id_unik)

        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(pasal_to_chunk_ids, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved pasal-to-chunk mapping to {mapping_path}")

