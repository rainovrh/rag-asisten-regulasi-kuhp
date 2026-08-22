#!/usr/bin/env python3
"""Run evaluation pipeline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.paths import DATASETS_DIR, LOGS_DIR, PROCESSED_DATA_DIR
from src.config.settings import settings
from src.utils.logger import setup_logger
from src.evaluation.golden_dataset import GoldenDataset
from src.evaluation.runner import EvaluationRunner
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.llm import LLMEngine
import json


def main() -> None:
    """Run evaluation."""
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Run RAGAS evaluation (Faithfulness, Answer Relevance)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path",
    )
    args = parser.parse_args()
    
    setup_logger("INFO")
    
    # Load corpus
    corpus_path = settings.processed_corpus_path
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    
    # Initialize components
    bm25 = BM25Retriever(corpus_data=corpus_data)
    dense = DenseRetriever(index_path=settings.faiss_index_path)
    retriever = HybridRetriever(bm25, dense)
    llm = LLMEngine()
    
    # Load dataset
    dataset_path = settings.golden_dataset_path
    dataset = GoldenDataset(dataset_path)
    
    # Load pasal mapping for evaluation
    pasal_mapping_path = PROCESSED_DATA_DIR / "pasal_to_chunks.json"
    
    # Run evaluation
    runner = EvaluationRunner(
        retriever,
        llm,
        dataset,
        pasal_mapping_path=pasal_mapping_path,
    )
    output_path = args.output or (LOGS_DIR / "evaluation_results.csv")
    results = runner.run(output_path, run_ragas=args.ragas)
    
    print(f"\nEvaluation complete. Results saved to {output_path}")
    print(f"Total scenarios: {len(results)}")


if __name__ == "__main__":
    main()
