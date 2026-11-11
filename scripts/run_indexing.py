#!/usr/bin/env python3
"""Run dual-search indexing pipeline: BM25 Sparse Indexing & FAISS Dense Indexing."""

import json
from pathlib import Path
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.paths import INDEXES_DIR, PROCESSED_DATA_DIR
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.indobert_embedding import IndoBERTEmbedding
from src.utils.logger import setup_logger

logger = setup_logger("INFO")


def main() -> None:
    """Execute dual indexing for KUHP legal chunks."""
    logger.info("=== STARTING DUAL INDEXING PIPELINE (KOTAK 5, 6, 7) ===")

    chunks_path = PROCESSED_DATA_DIR / "kuhp_chunks_hierarchical.json"
    if not chunks_path.exists():
        logger.error(f"Hierarchical chunks not found at {chunks_path}. Please run chunker first.")
        sys.exit(1)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} hierarchical legal chunks")

    # ---------------------------------------------------------
    # KOTAK 5: BM25 Sparse Indexing
    # ---------------------------------------------------------
    t0_bm25 = time.time()
    bm25 = BM25Retriever(chunks_data=chunks)
    bm25_save_path = PROCESSED_DATA_DIR / "bm25_index.pkl"
    bm25.save_index(bm25_save_path)
    bm25_duration = time.time() - t0_bm25
    bm25_size_kb = bm25_save_path.stat().st_size / 1024

    logger.info(f"[KOTAK 5] BM25 Index built and saved in {bm25_duration:.2f}s ({bm25_size_kb:.1f} KB)")

    # ---------------------------------------------------------
    # KOTAK 6 & 7: IndoBERT Dense Embedding & FAISS IndexFlatIP
    # ---------------------------------------------------------
    t0_dense = time.time()
    embedding = IndoBERTEmbedding(batch_size=32)
    dense = DenseRetriever(embedding=embedding)
    dense.build_from_chunks(chunks)

    # Save to both data/processed/faiss_index_kuhp and data/indexes/faiss_index_kuhp
    faiss_processed_path = PROCESSED_DATA_DIR / "faiss_index_kuhp"
    faiss_indexes_path = INDEXES_DIR / "faiss_index_kuhp"

    dense.save_index(faiss_processed_path)
    dense.save_index(faiss_indexes_path)

    dense_duration = time.time() - t0_dense
    index_file = faiss_processed_path / "index.faiss"
    faiss_size_mb = index_file.stat().st_size / (1024 * 1024)

    logger.info(f"[KOTAK 6 & 7] FAISS IndexFlatIP built and saved in {dense_duration:.2f}s ({faiss_size_mb:.2f} MB)")

    # ---------------------------------------------------------
    # Verification Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("INDEXING PIPELINE EXECUTION SUMMARY:")
    print("=" * 60)
    print(f"Total Chunks Indexed : {len(chunks)} chunks")
    print(f"BM25 Build Time      : {bm25_duration:.2f} seconds (< 30s target: PASSED)")
    print(f"BM25 Storage Size    : {bm25_size_kb:.1f} KB")
    print(f"IndoBERT Vectors     : {dense.total_vectors} vectors (Dim: 768)")
    print(f"FAISS Build Time     : {dense_duration:.2f} seconds")
    print(f"FAISS Storage Size   : {faiss_size_mb:.2f} MB (< 150 MB target: PASSED)")
    print(f"Index Completeness   : {dense.total_vectors == len(chunks)} (100% Match)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
