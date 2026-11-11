"""OCR artifact cleaning for legal documents."""

import re

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OcrCleaner:
    """Clean OCR artifacts from extracted legal text."""
    
    REPLACEMENTS = [
        (r'(?i)[F|P][R|N]ESIDEN\s+[^\s]*\.?[A-Z\.]*[^\s]*\s+INDONESIA', ' '),
        (r'(?i)[F|P][R|N]ESIDEN\s+[^\s]*INDONESIA', ' '),
        (r'(?i)REPUBUK\s+INDONESIA', ' '),
        (r'(?i)REPUBUK', ' '),
        (r'(?i)REPUEUK', ' '),
        (r'(?i)REPUBL\b', ' '),
        (r'(?i)REPUBLIKIK', ' '),
        (r'\|K\s+INDONESIA', ' '),
        (r'\|K\s+REPUELIK\s+INDONESIA', ' '),
        (r'\|K\s+INOONESIA', ' '),
        (r'AT\s+REPUBUK\s+INDONESIA', ' '),
        (r'INOONESIA', 'INDONESIA'),
        (r'rl\s*ffitrEIEtrN\s+REPUBL', ' '),
        (r'\s*\.\.\.\s*[\.\s]*[A-Za-z\[\]\{\}]*trtr[A-Za-z\[\]\{\}]*\s*', ' '),
        (r'\{II\s*l-irfl\{rf:IrfilNlr\'trltrFlltr\s*', ' '),
        (r'l-irfl\{rf:IrfilNlr\'trltrFlltr', ' '),
        (r'\s*asal\s+\d+O\d+\.{0,2}\s*', ' '),
        (r'\bal\s+\d+\.{0,2}\s*', ' '),
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
        (r'presiden\s+rei[\'`]*uelik\s+tndonesia', 'republik indonesia'),
        (r'rei\'?uelik\s+tndonesia', 'republik indonesia'),
        (r'dalarn', 'dalam'),
        (r'dalam\s+menetapkan', 'dalam menetapkan'),
        (r';a\s+lf\*r\'o\s+p', ''),
        (r'\bf\s*r\'?o\s+p\b', ''),
        (r'p\s*\(\s*l\s*\)', 'p (1)'),
        (r'p\s*\(\s*(\d+)\s*\)', r'p (\1)'),
        (r'[\*;]\s*[a-z]\s*\*', ''),
        (r'[;:]\s*[a-z]\s+[a-z]\s+[a-z]', ''),
        (r'\b[;:][a-z]\s*', ' '),
        (r'\s*[;:]\s*[a-z]\s*', ' '),
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
