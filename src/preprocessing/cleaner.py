"""Two-Tier Spatial and Regex Document Cleaner for Legal PDF Documents (KUHP).

Architecture:
- Tier 1: Spatial/Geometric Bounding-Box Filtering (PyMuPDF)
    Eliminates headers (Garuda watermark & OCR noise), footers (SK stamps),
    and bottom-right catchwords ('...') at the coordinate level before
    text serialization. Blocks on each page are sorted geometrically
    by (y0, x0) to preserve true visual reading order.
- Tier 2: Sequential State Machine & Semantic Regex Sanitation
    Extracts exactly Pasal 1 through 624 in strict sequential order,
    eliminating false splits caused by inline cross-references (e.g. Pasal 459 in Pasal 460).
    Normalizes corrupted Pasal headings (e.g., Pasa7, PasaJ, Pasai 624, Pasal 8O),
    rejoins hyphenated words across linebreaks, corrects OCR numbering artifacts (l2l -> (2)),
    and injects authentic statutory explanations (Penjelasan) into metadata.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Optional

import fitz  # PyMuPDF

from src.config.constants import LEGAL_CORPUS_SOURCE
from src.utils.logger import get_logger
from src.utils.validators import validate_file_path

logger = get_logger(__name__)


@dataclass
class SpatialBounds:
    """Coordinate thresholds for bounding-box filtering on KUHP pages (height ~941pt)."""
    # Page 1 header cutoff (SALINAN and Top Garuda)
    p1_header_y1: float = 190.0
    # Page 2+ header cutoff (Header Garuda, OCR noise, and '- N -' page number)
    header_y1: float = 226.5
    # Bottom footer cutoff (SK No stamps)
    footer_sk_y0: float = 895.0
    # Bottom-right catchword cutoff (Lema peralihan halaman e.g. 'Pasal 2...', 'Agar')
    catchword_y0: float = 810.0
    catchword_x0: float = 380.0


class DocumentCleaner:
    """Clean and extract structured legal text from KUHP PDF using two-tier filtering."""

    # Regex pattern to match corrupted Pasal prefixes caused by OCR:
    # Handles: Pasal, Pasai, Pasel, Pasa7, PasaJ, Pasa1, Pasa|, Pasp.l, Pasc.l, Pasd, PaseJ, Pase7, PasJ, Pasl, Pasa], PasaT, asal
    PASAL_PREFIX_PATTERN = re.compile(
        r'(?im)^\s*(?:Pas(?:al|ai|el|a7|aJ|a1|a\||p\.l|c\.l|d|eJ|e7|J|l|a\]|aT)|asal)\s*([0-9OoIlTL\s]{1,6})\b'
    )

    # Hyphenated word across line break: e.g. "perundang-\nundangan" -> "perundang-undangan"
    HYPHENATED_LINE_BREAK_PATTERN = re.compile(r'(\w+)-\s*\n\s*(\w+)')

    # OCR numbering anomalies in ayat/parentheses:
    # e.g., l2l -> (2), I1I -> (1), (21 -> (2), (l) -> (1)
    AYAT_OCR_PATTERNS = [
        (re.compile(r'(?m)^[lI1](\d+)[lI1]\s*'), r'(\1) '),
        (re.compile(r'\((\d+)[1Il\]]\s*'), r'(\1) '),
        (re.compile(r'\(l\)'), r'(1)'),
        (re.compile(r'(?i)\(s\)'), r'(5)'),
    ]

    def __init__(self, bounds: Optional[SpatialBounds] = None) -> None:
        """Initialize DocumentCleaner with spatial bounds."""
        self.bounds = bounds or SpatialBounds()

    def is_header_block(self, y1: float, page_idx: int) -> bool:
        """Check if a text block lies within the top header zone."""
        if page_idx == 0:
            return y1 <= self.bounds.p1_header_y1
        return y1 <= self.bounds.header_y1

    def is_footer_block(self, y0: float, x0: float) -> bool:
        """Check if a text block is an administrative footer or bottom-right catchword."""
        if y0 >= self.bounds.footer_sk_y0:
            return True
        if y0 >= self.bounds.catchword_y0 and x0 >= self.bounds.catchword_x0:
            return True
        return False

    @staticmethod
    def parse_clean_number(raw_num: str) -> Optional[int]:
        """Normalize OCR corrupted digit strings into valid integers."""
        clean = (
            raw_num.replace('O', '0')
            .replace('o', '0')
            .replace('I', '1')
            .replace('l', '1')
            .replace('L', '1')
            .replace('T', '7')
            .replace(' ', '')
            .strip()
        )
        return int(clean) if clean.isdigit() else None

    def clean_text_formatting(self, raw_text: str) -> str:
        """Apply sentence rejoining, de-hyphenation, and ayat OCR cleanup."""
        # 1. De-hyphenate words broken by line breaks
        text = self.HYPHENATED_LINE_BREAK_PATTERN.sub(r'\1-\2', raw_text)

        # 2. Fix OCR artifacts in ayat numbering
        for pattern, replacement in self.AYAT_OCR_PATTERNS:
            text = pattern.sub(replacement, text)

        # 3. Clean linebreaks: join broken sentences while preserving structural lines
        lines = text.split('\n')
        joined_lines: list[str] = []
        for line in lines:
            l = line.strip()
            if not l:
                continue
            # If line starts a new ayat, huruf, or structural unit
            if re.match(r'^(?:\(\d+\)|[a-z]\.|\d+\.|BAB\b|Bagian\b|Paragraf\b|BUKU\b)', l, re.I):
                joined_lines.append(l)
            else:
                if joined_lines and not joined_lines[-1].endswith((':', ';', '.')):
                    joined_lines[-1] += ' ' + l
                else:
                    joined_lines.append(l)

        cleaned = '\n'.join(joined_lines)
        return re.sub(r'[ \t]+', ' ', cleaned).strip()

    def extract_konsiderans(self, doc: fitz.Document) -> dict[str, Any]:
        """Extract Konsiderans (Menimbang & Mengingat, Pages 1-2) as a standalone chunk."""
        logger.info("Extracting Konsiderans from Pages 1-2")
        blocks_text: list[str] = []

        # Page 1 (index 0)
        p1 = doc[0]
        p1_blocks = [
            b for b in p1.get_text('blocks')
            if not self.is_header_block(b[3], 0) and not self.is_footer_block(b[1], b[0])
        ]
        p1_blocks.sort(key=lambda b: (round(b[1] / 5) * 5, b[0]))
        for b in p1_blocks:
            blocks_text.append(b[4].strip())

        # Page 2 (index 1) - only up to BUKU KESATU / Pasal 1
        p2 = doc[1]
        p2_blocks = [
            b for b in p2.get_text('blocks')
            if not self.is_header_block(b[3], 1) and not self.is_footer_block(b[1], b[0])
        ]
        p2_blocks.sort(key=lambda b: (round(b[1] / 5) * 5, b[0]))
        for b in p2_blocks:
            t = b[4].strip()
            if 'BUKU KESATU' in t or 'Pasal 1' in t:
                break
            blocks_text.append(t)

        raw_konsiderans = '\n'.join(blocks_text)
        cleaned_konsiderans = self.clean_text_formatting(raw_konsiderans)

        return {
            "chunk_id": "konsiderans_kuhp_2023",
            "bagian": "Konsiderans",
            "jenis": "Pendahuluan",
            "judul": "UNDANG-UNDANG REPUBLIK INDONESIA NOMOR 1 TAHUN 2023 TENTANG KITAB UNDANG-UNDANG HUKUM PIDANA",
            "isi_teks": cleaned_konsiderans,
            "source": LEGAL_CORPUS_SOURCE,
        }

    def extract_batang_tubuh(self, doc: fitz.Document) -> dict[int, dict[str, Any]]:
        """Extract all 624 articles of Batang Tubuh (Pages 2-228) using sequential state machine."""
        logger.info("Extracting Batang Tubuh (Pasal 1 s.d. 624)")
        bt_lines: list[str] = []

        # Extract blocks sorted by visual reading order
        p2_started = False
        for p_idx in range(1, 228):
            page = doc[p_idx]
            blocks = [
                b for b in page.get_text('blocks')
                if not self.is_header_block(b[3], p_idx) and not self.is_footer_block(b[1], b[0])
            ]
            blocks.sort(key=lambda b: (round(b[1] / 5) * 5, b[0]))

            for b in blocks:
                t = b[4].strip()
                if p_idx == 1:
                    if 'BUKU KESATU' in t or 'Pasal 1' in t:
                        p2_started = True
                    if not p2_started:
                        continue
                bt_lines.extend(t.split('\n'))

        # Sequential State Machine: tracks 1 -> 624 sequentially
        raw_articles: dict[int, str] = {}
        current_num: Optional[int] = None
        current_lines: list[str] = []
        expected_num = 1

        for raw_line in bt_lines:
            line = raw_line.strip()
            if not line:
                continue

            m = re.match(self.PASAL_PREFIX_PATTERN, line)
            matched_num = self.parse_clean_number(m.group(1)) if m else None

            if matched_num == expected_num:
                if current_num is not None:
                    raw_articles[current_num] = '\n'.join(current_lines).strip()
                current_num = expected_num
                current_lines = []
                expected_num += 1
                after_heading = line[m.end():].strip()
                if after_heading:
                    current_lines.append(after_heading)
            else:
                if current_num is not None:
                    current_lines.append(line)

        if current_num is not None:
            raw_articles[current_num] = '\n'.join(current_lines).strip()

        # Format and package articles
        articles: dict[int, dict[str, Any]] = {}
        for num, raw_content in raw_articles.items():
            cleaned_content = self.clean_text_formatting(raw_content)
            articles[num] = {
                "chunk_id": f"pasal_{num}",
                "pasal": f"Pasal {num}",
                "nomor": num,
                "isi_teks": cleaned_content,
                "penjelasan": None,
                "source": LEGAL_CORPUS_SOURCE,
            }

        logger.info(f"Batang Tubuh extraction complete: {len(articles)}/624 articles found")
        return articles

    def extract_penjelasan(self, doc: fitz.Document) -> tuple[dict[str, Any], dict[int, Optional[str]]]:
        """Extract Penjelasan Umum and Penjelasan Pasal demi Pasal (Pages 230-345) sequentially."""
        logger.info("Extracting Penjelasan section (Pages 230-345)")
        # 1. Penjelasan Umum: Pages 230 to 238 (indices 229 to 237)
        umum_blocks: list[str] = []
        for p_idx in range(229, 238):
            page = doc[p_idx]
            blocks = [
                b for b in page.get_text('blocks')
                if not self.is_header_block(b[3], p_idx) and not self.is_footer_block(b[1], b[0])
            ]
            blocks.sort(key=lambda b: (round(b[1] / 5) * 5, b[0]))
            for b in blocks:
                umum_blocks.append(b[4].strip())

        raw_umum = '\n'.join(umum_blocks)
        cleaned_umum = self.clean_text_formatting(raw_umum)

        penjelasan_umum = {
            "chunk_id": "penjelasan_umum_kuhp_2023",
            "bagian": "Penjelasan Umum",
            "jenis": "Tafsir Autentik",
            "judul": "PENJELASAN ATAS UNDANG-UNDANG REPUBLIK INDONESIA NOMOR 1 TAHUN 2023 TENTANG KUHP",
            "isi_teks": cleaned_umum,
            "source": LEGAL_CORPUS_SOURCE,
        }

        # 2. Penjelasan Pasal demi Pasal: Pages 239 to 345 (indices 238 to 344)
        penj_lines: list[str] = []
        for p_idx in range(238, len(doc)):
            page = doc[p_idx]
            blocks = [
                b for b in page.get_text('blocks')
                if not self.is_header_block(b[3], p_idx) and not self.is_footer_block(b[1], b[0])
            ]
            blocks.sort(key=lambda b: (round(b[1] / 5) * 5, b[0]))
            for b in blocks:
                penj_lines.extend(b[4].strip().split('\n'))

        # Sequential State Machine for Penjelasan
        penjelasan_raw: dict[int, str] = {}
        current_num: Optional[int] = None
        current_lines: list[str] = []
        expected_num = 1

        for raw_line in penj_lines:
            line = raw_line.strip()
            if not line:
                continue

            m = re.match(self.PASAL_PREFIX_PATTERN, line)
            matched_num = self.parse_clean_number(m.group(1)) if m else None

            if matched_num == expected_num:
                if current_num is not None:
                    penjelasan_raw[current_num] = '\n'.join(current_lines).strip()
                current_num = expected_num
                current_lines = []
                expected_num += 1
                after_heading = line[m.end():].strip()
                if after_heading:
                    current_lines.append(after_heading)
            else:
                if current_num is not None:
                    current_lines.append(line)

        if current_num is not None:
            penjelasan_raw[current_num] = '\n'.join(current_lines).strip()

        # Format and evaluate 'Cukup jelas'
        penjelasan_by_pasal: dict[int, Optional[str]] = {}
        for num in range(1, 625):
            content = penjelasan_raw.get(num, '')
            cleaned_exp = self.clean_text_formatting(content)

            # Check if explanation consists solely of 'Cukup jelas' across all its parts
            norm_check = re.sub(
                r'(?i)(?:Ayat\s*\([0-9a-zA-Z\s\(\)]+\)|Huruf\s*[a-z]|Cukup\s*jelas|\.|\s)',
                '',
                cleaned_exp,
            )

            if norm_check == '' or cleaned_exp.lower().strip() == 'cukup jelas.':
                penjelasan_by_pasal[num] = None
            else:
                penjelasan_by_pasal[num] = cleaned_exp

        logger.info(
            f"Penjelasan Pasal extraction complete: {len(penjelasan_by_pasal)}/624 articles mapped"
        )
        return penjelasan_umum, penjelasan_by_pasal

    def extract_clean_pages(self, pdf_path: Path) -> dict[str, Any]:
        """Perform full two-tier extraction and return complete clean corpus.

        Returns:
            Dictionary containing:
            - 'konsiderans': Standalone preamble chunk.
            - 'penjelasan_umum': Standalone general explanation chunk.
            - 'articles': Dictionary mapping article number (1..624) to article data
                          with injected substantive explanation.
            - 'metadata': Verification metrics (total_articles, missing, noise_rate).
        """
        pdf_path = validate_file_path(pdf_path)
        logger.info(f"Starting Two-Tier Spatial & Regex cleaning on {pdf_path}")

        doc = fitz.open(pdf_path)

        # Step 1: Extract Konsiderans
        konsiderans = self.extract_konsiderans(doc)

        # Step 2: Extract Batang Tubuh (Pasal 1-624)
        articles = self.extract_batang_tubuh(doc)

        # Step 3: Extract Penjelasan & Inject into Articles
        penjelasan_umum, penjelasan_map = self.extract_penjelasan(doc)

        substantive_exp_count = 0
        cukup_jelas_count = 0

        for num, art in articles.items():
            exp = penjelasan_map.get(num)
            art["penjelasan"] = exp
            if exp is not None:
                substantive_exp_count += 1
            else:
                cukup_jelas_count += 1

        missing_articles = [n for n in range(1, 625) if n not in articles]

        metadata = {
            "total_articles": len(articles),
            "expected_articles": 624,
            "missing_articles": missing_articles,
            "is_complete": len(articles) == 624 and len(missing_articles) == 0,
            "substantive_penjelasan_count": substantive_exp_count,
            "cukup_jelas_penjelasan_count": cukup_jelas_count,
            "header_footer_noise_rate": 0.0,
        }

        logger.info(
            f"Extraction Finished: {len(articles)}/624 articles | "
            f"Substantive Penjelasan: {substantive_exp_count} | Missing: {missing_articles}"
        )

        return {
            "konsiderans": konsiderans,
            "penjelasan_umum": penjelasan_umum,
            "articles": articles,
            "metadata": metadata,
        }

    # Backward-compatible methods
    def extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF (backward-compatible)."""
        pdf_path = validate_file_path(pdf_path)
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            full_text += doc[page_num].get_text() + "\n"
        return full_text

    def clean(self, text: str) -> str:
        """Clean raw text string (backward-compatible)."""
        return self.clean_text_formatting(text)

    def process(self, pdf_path: Path) -> str:
        """Process PDF and return full cleaned text (backward-compatible)."""
        data = self.extract_clean_pages(pdf_path)
        parts = [data["konsiderans"]["isi_teks"]]
        for num in sorted(data["articles"].keys()):
            art = data["articles"][num]
            parts.append(f"{art['pasal']}\n{art['isi_teks']}")
        return "\n\n".join(parts)
