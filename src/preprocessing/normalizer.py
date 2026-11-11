"""Text normalization and query validation utilities for Indonesian legal RAG (Blok 2, Kotak 8 & 9)."""

from dataclasses import dataclass
import re
import time
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QueryValidationResult:
    """Result of user legal case inquiry validation (Kotak 8)."""

    is_valid: bool
    sanitized_query: str
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    word_count: int = 0


class UserQueryValidator:
    """Validator for user legal case inquiries (Kotak 8).
    
    Guarantees:
    - Zero crash / unhandled exceptions on malicious or boundary inputs.
    - Clean rejection of empty/whitespace queries.
    - Safe handling and truncation of queries exceeding word limits.
    - Neutralization of script tags, SQL injection snippets, and null bytes.
    """

    MAX_WORDS: int = 500
    MAX_CHAR_LENGTH: int = 4000

    # Common injection and dangerous patterns
    DANGEROUS_PATTERNS = [
        re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"<\/?(?:script|iframe|object|embed|applet|style).*?>", re.IGNORECASE),
        re.compile(r"(--\s*$|;\s*(?:DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT)\b)", re.IGNORECASE),
        re.compile(r"['\"]\s*OR\s*['\"]?1['\"]?\s*=\s*['\"]?1", re.IGNORECASE),
        re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"),  # Control characters
    ]

    @classmethod
    def validate(cls, raw_query: Optional[str]) -> QueryValidationResult:
        """Validate and sanitize user inquiry safely.

        Args:
            raw_query: Raw user input string.

        Returns:
            QueryValidationResult indicating validity, sanitized text, and error/warning.
        """
        # 1. Null or non-string check
        if raw_query is None:
            return QueryValidationResult(
                is_valid=False,
                sanitized_query="",
                error_message="Kueri tidak boleh kosong (None).",
                word_count=0,
            )

        if not isinstance(raw_query, str):
            return QueryValidationResult(
                is_valid=False,
                sanitized_query="",
                error_message="Format kueri harus berupa teks string.",
                word_count=0,
            )

        # 2. Whitespace-only check
        stripped = raw_query.strip()
        if not stripped:
            return QueryValidationResult(
                is_valid=False,
                sanitized_query="",
                error_message="Kueri tidak boleh hanya berupa spasi kosong.",
                word_count=0,
            )

        # 3. Neutralize dangerous patterns (XSS, SQLi, Control chars)
        sanitized = stripped
        for pattern in cls.DANGEROUS_PATTERNS:
            sanitized = pattern.sub(" ", sanitized)

        # Re-strip and normalize whitespace after sanitization
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        if not sanitized:
            return QueryValidationResult(
                is_valid=False,
                sanitized_query="",
                error_message="Kueri hanya memuat karakter tidak valid atau skrip berbahaya.",
                word_count=0,
            )

        # 4. Word count & character length check
        words = sanitized.split()
        word_count = len(words)
        warning: Optional[str] = None

        if word_count > cls.MAX_WORDS:
            words = words[: cls.MAX_WORDS]
            sanitized = " ".join(words)
            warning = f"Kueri melebihi batas {cls.MAX_WORDS} kata dan telah dipotong secara aman."
            word_count = cls.MAX_WORDS

        if len(sanitized) > cls.MAX_CHAR_LENGTH:
            sanitized = sanitized[: cls.MAX_CHAR_LENGTH].rsplit(" ", 1)[0]
            warning = (warning or "") + f" Kueri dipangkas pada batas {cls.MAX_CHAR_LENGTH} karakter."

        return QueryValidationResult(
            is_valid=True,
            sanitized_query=sanitized,
            error_message=None,
            warning_message=warning,
            word_count=word_count,
        )


class LegalQueryNormalizer:
    """Preprocesses and normalizes legal queries (Kotak 9).
    
    Guarantees:
    - Normalization execution time < 10 ms.
    - 100% preservation of legal keywords and normative operators.
    - Accurate expansion of Indonesian statutory abbreviations.
    - Retention of article and clause numbering syntax (e.g. Pasal 1 ayat (1)).
    """

    # Legal abbreviation expansion dictionary
    ABBREVIATIONS = {
        r"\buu\b": "undang-undang",
        r"\bkuhp\b": "kitab undang-undang hukum pidana",
        r"\buud\b": "undang-undang dasar",
        r"\bpsl\.?\b": "pasal",
        r"\bps\.?\b": "pasal",
        r"\bayt\.?\b": "ayat",
        r"\bperpu\b": "peraturan pemerintah pengganti undang-undang",
        r"\bpp\b": "peraturan pemerintah",
        r"\bwni\b": "warga negara indonesia",
        r"\bwna\b": "warga negara asing",
        r"\bnkri\b": "negara kesatuan republik indonesia",
        r"\bham\b": "hak asasi manusia",
    }

    def __init__(
        self,
        lowercase: bool = True,
        expand_abbreviations: bool = True,
    ) -> None:
        """Initialize legal query normalizer."""
        self.lowercase = lowercase
        self.expand_abbreviations = expand_abbreviations

        # Compile regex patterns once for maximum performance (< 10 ms)
        self._compiled_abbreviations = [
            (re.compile(pattern, re.IGNORECASE), replacement)
            for pattern, replacement in self.ABBREVIATIONS.items()
        ]
        self._whitespace_pattern = re.compile(r"\s+")
        # Keep letters, digits, standard punctuation, and parentheses for ayat e.g. (1)
        self._noise_symbols_pattern = re.compile(r"[^\w\s\(\)\.,\-\?\/]")

    def normalize(self, query: str) -> str:
        """Apply full legal query normalization pipeline.

        Args:
            query: Input user query string.

        Returns:
            Cleaned and normalized query string.
        """
        if not query:
            return ""

        # 1. Clean unusual non-legal symbols
        text = self._noise_symbols_pattern.sub(" ", query)

        # 2. Normalize whitespace
        text = self._whitespace_pattern.sub(" ", text).strip()

        # 3. Case folding
        if self.lowercase:
            text = text.lower()

        # 4. Expand legal abbreviations
        if self.expand_abbreviations:
            for pattern, replacement in self._compiled_abbreviations:
                text = pattern.sub(replacement, text)

        # 5. Final whitespace cleanup
        text = self._whitespace_pattern.sub(" ", text).strip()
        return text

    def measure_latency(self, query: str) -> tuple[str, float]:
        """Normalize query and return normalized text alongside execution time in milliseconds."""
        t0 = time.perf_counter()
        normalized = self.normalize(query)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return normalized, duration_ms


# Maintain backward compatibility with TextNormalizer alias
TextNormalizer = LegalQueryNormalizer
