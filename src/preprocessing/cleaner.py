"""Document cleaning and text extraction."""

import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from src.utils.logger import get_logger
from src.utils.validators import validate_file_path
from src.config.constants import LEGAL_CORPUS_SOURCE

logger = get_logger(__name__)


class DocumentCleaner:
    """Clean and extract text from legal PDF documents."""
    
    # Regex patterns for cleaning
    HEADER_PATTERNS = [
        re.compile(r'PRESIDEN\s*REPUBLIK\s*INDONESIA', re.IGNORECASE),
        re.compile(r'SALINAN', re.IGNORECASE),
        re.compile(r'SK\s+No\s+\d+[A-Z\s]*', re.IGNORECASE),
    ]
    
    PAGE_NUMBER_PATTERN = re.compile(r'\n-\s*\d+\s*-\n')
    WHITESPACE_PATTERN = re.compile(r'\n\s*\n')
    
    def __init__(self, remove_page_numbers: bool = True) -> None:
        """Initialize document cleaner.
        
        Args:
            remove_page_numbers: Whether to remove page numbers.
        """
        self.remove_page_numbers = remove_page_numbers
    
    def extract_text(self, pdf_path: Path) -> str:
        """Extract raw text from PDF.
        
        Args:
            pdf_path: Path to PDF file.
        
        Returns:
            Extracted text content.
        """
        pdf_path = validate_file_path(pdf_path)
        logger.info(f"Extracting text from {pdf_path}")
        
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text()
            full_text += page_text + "\n"
        
        logger.info(f"Extracted {len(full_text)} characters from {len(doc)} pages")
        return full_text
    
    def clean(self, text: str) -> str:
        """Clean extracted text by removing noise.
        
        Args:
            text: Raw extracted text.
        
        Returns:
            Cleaned text.
        """
        logger.debug("Cleaning extracted text")
        
        # Remove headers/watermarks
        for pattern in self.HEADER_PATTERNS:
            text = pattern.sub('', text)
        
        # Remove page numbers
        if self.remove_page_numbers:
            text = self.PAGE_NUMBER_PATTERN.sub('\n', text)
        
        # Normalize whitespace
        text = self.WHITESPACE_PATTERN.sub('\n', text)
        
        logger.debug(f"Cleaned text length: {len(text)} characters")
        return text
    
    def process(self, pdf_path: Path) -> str:
        """Extract and clean text from PDF.
        
        Args:
            pdf_path: Path to PDF file.
        
        Returns:
            Cleaned text content.
        """
        raw_text = self.extract_text(pdf_path)
        return self.clean(raw_text)
