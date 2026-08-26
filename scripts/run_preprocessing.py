#!/usr/bin/env python3
"""Run preprocessing pipeline."""

import argparse
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.config.settings import settings
from src.preprocessing.cleaner import DocumentCleaner
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.ocr_cleaner import OcrCleaner
from src.preprocessing.chunker import PasalSegmenter, SemanticChunker, FixedSizeChunker
from src.utils.logger import setup_logger


def _load_pasal_segments_from_backup() -> list[tuple[str, str]]:
    """Load pasal segments from backup pasal-level corpus.
    
    Returns:
        List of (pasal_id, pasal_text) tuples.
    """
    backup_path = PROCESSED_DATA_DIR / "kuhp_bersih.json.bak"
    if not backup_path.exists():
        raise FileNotFoundError(
            f"Backup corpus not found at {backup_path}. "
            "Cannot use pasal-level segmentation without backup."
        )
    
    with open(backup_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    segments = []
    for key, text in data.items():
        pasal_id = key.lower()
        segments.append((pasal_id, text))
    
    return segments


def _clean_text(text: str) -> str:
    """Apply full cleaning pipeline to text.
    
    Args:
        text: Raw text.
    
    Returns:
        Cleaned text.
    """
    cleaner = DocumentCleaner()
    cleaned = cleaner.clean(text)
    ocr_cleaner = OcrCleaner()
    cleaned = ocr_cleaner.clean(cleaned)
    normalizer = TextNormalizer()
    normalized = normalizer.normalize(cleaned)
    return normalized


def main() -> None:
    """Run preprocessing pipeline."""
    parser = argparse.ArgumentParser(description="Preprocess KUHP corpus")
    parser.add_argument(
        "--method",
        choices=["semantic", "fixed", "both"],
        default="semantic",
        help="Chunking method: semantic (proposed), fixed (baseline), or both",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in tokens for fixed-size chunking",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap in tokens for fixed-size chunking",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for semantic chunking",
    )
    parser.add_argument(
        "--use-backup-segments",
        action="store_true",
        help="Use backup pasal-level corpus for segmentation (recommended)",
    )
    args = parser.parse_args()

    setup_logger("INFO")
    
    pdf_path = RAW_DATA_DIR / "KUHP BARU UU Nomor 1 Tahun 2023.pdf"
    
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)
    
    # Step 1: Get pasal segments
    if args.use_backup_segments:
        print("Using backup pasal-level corpus for segmentation")
        raw_segments = _load_pasal_segments_from_backup()
        segments = []
        for pasal_id, text in raw_segments:
            cleaned_text = _clean_text(text)
            if cleaned_text.strip():
                segments.append((pasal_id, cleaned_text))
        print(f"Loaded {len(segments)} pasal segments from backup")
    else:
        print("Using line-anchored pasal segmentation on cleaned document")
        cleaner = DocumentCleaner()
        cleaned_text = cleaner.process(pdf_path)
        ocr_cleaner = OcrCleaner()
        cleaned_text = ocr_cleaner.clean(cleaned_text)
        segmenter = PasalSegmenter(line_anchored=True)
        raw_segments = segmenter.segment(cleaned_text)
        normalizer = TextNormalizer()
        segments = []
        for pasal_id, text in raw_segments:
            normalized_text = normalizer.normalize(text)
            if normalized_text.strip():
                segments.append((pasal_id, normalized_text))
        print(f"Segmented into {len(segments)} pasal segments")
    
    # Step 2: Chunk
    methods = []
    if args.method == "semantic":
        methods.append(("semantic", SemanticChunker(threshold=args.threshold)))
    elif args.method == "fixed":
        methods.append(("fixed", FixedSizeChunker(chunk_size=args.chunk_size, overlap=args.overlap)))
    elif args.method == "both":
        methods.append(("semantic", SemanticChunker(threshold=args.threshold)))
        methods.append(("fixed", FixedSizeChunker(chunk_size=args.chunk_size, overlap=args.overlap)))
    
    for method_name, chunker in methods:
        print(f"\n--- Running {method_name} chunking ---")
        
        if isinstance(chunker, SemanticChunker):
            chunks = chunker.chunk_segments(segments)
        else:
            chunks = chunker.chunk_segments(segments)
        
        # Save corpus (chunk_id -> text)
        if method_name == "semantic":
            output_path = PROCESSED_DATA_DIR / "kuhp_bersih.json"
        else:
            output_path = PROCESSED_DATA_DIR / f"kuhp_bersih_{method_name}.json"
        
        corpus = {chunk.chunk_id: chunk.text for chunk in chunks}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        
        # Save metadata (chunk_id -> pasal_refs, source, char_count, token_count)
        metadata_path = output_path.with_suffix(".metadata.json")
        metadata = {
            chunk.chunk_id: {
                "pasal_refs": chunk.pasal_refs or [],
                "source": chunk.source,
                "char_count": chunk.char_count,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(chunks)} chunks to {output_path}")
        print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
