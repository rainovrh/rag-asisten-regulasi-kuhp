"""OCR artifact cleaning for legal documents."""

import re

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OcrCleaner:
    """Clean OCR artifacts from extracted legal text."""
    
    REPLACEMENTS = [
        (r'FRESIDEN\s+[^\s]*\.?[A-Z\.]*[^\s]*\s+INDONESIA', 'REPUBLIK INDONESIA'),
        (r'FRESIDEN\s+[^\s]*INDONESIA', 'REPUBLIK INDONESIA'),
        (r'FRESIDEN\s+REPUBUK\s+INDONESIA', 'REPUBLIK INDONESIA'),
        (r'FRESIDEN\s+REI\'UBUK\s+INDONESIA', 'REPUBLIK INDONESIA'),
        (r'FRESIDEN\s+REPUELIK\s+INDONESIA', 'REPUBLIK INDONESIA'),
        (r'PNES!DEN\s+REPTIEIJK\s+INDONESIA', 'REPUBLIK INDONESIA'),
        (r'PNES!DEN\s+REPTIEIJK', 'REPUBLIK'),
        (r'\|K\s+INDONESIA', 'REPUBLIK INDONESIA'),
        (r'\|K\s+REPUELIK\s+INDONESIA', 'REPUBLIK INDONESIA'),
        (r'\|K\s+INOONESIA', 'REPUBLIK INDONESIA'),
        (r'AT\s+REPUBUK\s+INDONESIA', 'AT REPUBLIK INDONESIA'),
        (r'REPUBUK', 'REPUBLIK'),
        (r'REPUEUK', 'REPUBLIK'),
        (r'REPUBL', 'REPUBLIK'),
        (r'INOONESIA', 'INDONESIA'),
        (r'REPUBLIKIK', 'REPUBLIK'),
        (r'rl\s*ffitrEIEtrN\s+REPUBL', ' '),
        (r'\s*\.\.\.\s*[\.\s]*[A-Za-z\[\]\{\}]*trtr[A-Za-z\[\]\{\}]*\s*', ' '),
        (r'\{II\s*l-irfl\{rf:IrfilNlr\'trltrFlltr\s*', ' '),
        (r'l-irfl\{rf:IrfilNlr\'trltrFlltr', ' '),
        (r'\s*asal\s+\d+O\d+\.{0,2}\s*', ' '),
        (r'\s*asal\s+\d+\.{0,2}\s*', ' '),
        (r'Ayat\s*\((l)\)', 'Ayat (1)'),
        (r'Ayat(l)', 'Ayat (1)'),
        (r'Ayatl(\d+)l', r'Ayat (\1)'),
        (r'Pasal(\d+)\s', r'Pasal \1 '),
        (r'Pasel(\d+)', r'Pasal \1'),
        (r'SK\s+No\s+l(\d+)l(\d+)l\s*([A-Z])', r'SK No. \g<1>\g<2>-\g<3>'),
        (r'Tahwr', 'Tahun'),
        (r'Tahw\s', 'Tahun '),
        (r'lgtaFi\s+', ' '),
        (r'daar\s+', ' '),
        (r'\.{4,}', '.'),
        (r'  +', ' '),
    ]
    
    def clean(self, text: str) -> str:
        """Apply OCR artifact cleaning.
        
        Args:
            text: Input text with potential OCR artifacts.
        
        Returns:
            Cleaned text.
        """
        original_length = len(text)
        
        for pattern, replacement in self.REPLACEMENTS:
            text = re.sub(pattern, replacement, text)
        
        text = text.strip()
        
        if len(text) != original_length:
            logger.debug(f"OCR cleaning changed text length: {original_length} -> {len(text)}")
        
        return text
