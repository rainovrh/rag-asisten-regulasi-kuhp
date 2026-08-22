#!/usr/bin/env python3
"""Run preprocessing pipeline."""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.config.settings import settings
from src.preprocessing.cleaner import DocumentCleaner
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.ocr_cleaner import OcrCleaner
from src.preprocessing.chunker import SemanticChunker
from src.utils.logger import setup_logger


def main() -> None:
    """Run preprocessing pipeline."""
    setup_logger("INFO")
    
    pdf_path = RAW_DATA_DIR / "KUHP BARU UU Nomor 1 Tahun 2023.pdf"
    output_path = PROCESSED_DATA_DIR / "kuhp_bersih.json"
    
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)
    
    # Step 1: Extract and clean
    cleaner = DocumentCleaner()
    cleaned_text = cleaner.process(pdf_path)
    
    # Step 1.5: OCR artifact cleaning
    ocr_cleaner = OcrCleaner()
    cleaned_text = ocr_cleaner.clean(cleaned_text)
    
    # Step 2: Normalize
    normalizer = TextNormalizer()
    normalized_text = normalizer.normalize(cleaned_text)
    
    # Step 3: Chunk
    chunker = SemanticChunker()
    chunks = chunker.chunk(normalized_text)
    
    # Step 4: Save
    import json
    corpus = {}
    for chunk in chunks:
        corpus[chunk.chunk_id] = chunk.text
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    main()
