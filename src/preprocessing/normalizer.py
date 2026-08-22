"""Text normalization utilities."""

import re
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextNormalizer:
    """Normalize text for Indonesian legal documents."""
    
    # Abbreviation expansions
    ABBREVIATIONS = {
        "uu": "undang-undang",
        "kuhp": "kitab undang-undang hukum pidana",
        "kup": "kesehatan",
        "pidana": "pidana",
    }
    
    def __init__(
        self,
        lowercase: bool = True,
        expand_abbreviations: bool = True,
    ) -> None:
        """Initialize text normalizer.
        
        Args:
            lowercase: Convert text to lowercase.
            expand_abbreviations: Expand common abbreviations.
        """
        self.lowercase = lowercase
        self.expand_abbreviations = expand_abbreviations
    
    def normalize(self, text: str) -> str:
        """Apply all normalization steps.
        
        Args:
            text: Input text.
        
        Returns:
            Normalized text.
        """
        text = self._normalize_whitespace(text)
        
        if self.lowercase:
            text = self._lowercase(text)
        
        if self.expand_abbreviations:
            text = self._expand_abbreviations(text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def _lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common Indonesian legal abbreviations."""
        # Simple word-boundary replacement
        for abbr, expansion in self.ABBREVIATIONS.items():
            pattern = re.compile(r'\b' + re.escape(abbr) + r'\b', re.IGNORECASE)
            text = pattern.sub(expansion, text)
        
        return text
