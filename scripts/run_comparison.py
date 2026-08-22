#!/usr/bin/env python3
"""Run comparison evaluation across 4 scenarios."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.paths import DATASETS_DIR, LOGS_DIR, PROCESSED_DATA_DIR, INDEXES_DIR
from src.config.settings import settings
from src.utils.logger import setup_logger
from src.evaluation.golden_dataset import GoldenDataset
from src.evaluation.runner import EvaluationRunner
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.llm import LLMEngine
import pandas as pd


def run_scenario(
    name: str,
    chunking_method: str,
    retrieval_method: str,
    corpus_path: Path,
    index_path: Path,
    dataset: GoldenDataset,
    pasal_mapping_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Run a single evaluation scenario.
    
    Args:
        name: Scenario name.
        chunking_method: 'semantic' or 'fixed'.
        retrieval_method: 'hybrid', 'bm25', 'dense', or 'none'.
        corpus_path: Path to corpus JSON.
        index_path: Path to FAISS index.
        dataset: Golden dataset.
        pasal_mapping_path: Path to pasal mapping.
        output_dir: Directory to save results.
    
    Returns:
        DataFrame with results.
    """
    print(f"\n{'='*70}")
    print(f"Scenario: {name}")
    print(f"  Chunking: {chunking_method}")
    print(f"  Retrieval: {retrieval_method}")
    print(f"{'='*70}")
    
    # Load corpus
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    
    # Initialize retrievers
    bm25 = BM25Retriever(corpus_data=corpus_data)
    dense = DenseRetriever(index_path=index_path)
    
    # Configure retriever mode
    use_bm25 = retrieval_method in ["bm25", "hybrid"]
    use_dense = retrieval_method in ["dense", "hybrid"]
    
    retriever = HybridRetriever(
        bm25,
        dense,
        use_bm25=use_bm25,
        use_dense=use_dense,
    )
    
    llm = LLMEngine()
    
    # Run evaluation
    runner = EvaluationRunner(
        retriever,
        llm,
        dataset,
        pasal_mapping_path=pasal_mapping_path,
    )
    
    output_path = output_dir / f"{name.replace(' ', '_').lower()}_results.csv"
    df = runner.run(output_path)
    
    # Add scenario metadata
    df["Scenario"] = name
    df["Chunking_Method"] = chunking_method
    df["Retrieval_Method"] = retrieval_method
    
    return df


def main() -> None:
    """Run comparison across all scenarios."""
    setup_logger("INFO")
    
    output_dir = LOGS_DIR / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset_path = settings.golden_dataset_path
    dataset = GoldenDataset(dataset_path)
    pasal_mapping_path = PROCESSED_DATA_DIR / "pasal_to_chunks.json"
    
    # Define scenarios
    scenarios = [
        {
            "name": "A_Baseline_Murni",
            "chunking": "fixed",
            "retrieval": "bm25",
            "corpus": PROCESSED_DATA_DIR / "kuhp_bersih_fixed.json",
            "index": INDEXES_DIR / "faiss_index_kuhp",
        },
        {
            "name": "B_Semantik_Tunggal",
            "chunking": "semantic",
            "retrieval": "dense",
            "corpus": PROCESSED_DATA_DIR / "kuhp_bersih.json",
            "index": INDEXES_DIR / "faiss_index_kuhp",
        },
        {
            "name": "C_Leksikal_Tunggal",
            "chunking": "fixed",
            "retrieval": "bm25",
            "corpus": PROCESSED_DATA_DIR / "kuhp_bersih_fixed.json",
            "index": INDEXES_DIR / "faiss_index_kuhp",
        },
        {
            "name": "D_Sistem_Usulan",
            "chunking": "semantic",
            "retrieval": "hybrid",
            "corpus": PROCESSED_DATA_DIR / "kuhp_bersih.json",
            "index": INDEXES_DIR / "faiss_index_kuhp",
        },
    ]
    
    all_results = []
    for scenario in scenarios:
        df = run_scenario(
            name=scenario["name"],
            chunking_method=scenario["chunking"],
            retrieval_method=scenario["retrieval"],
            corpus_path=scenario["corpus"],
            index_path=scenario["index"],
            dataset=dataset,
            pasal_mapping_path=pasal_mapping_path,
            output_dir=output_dir,
        )
        all_results.append(df)
    
    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = output_dir / "comparison_results.csv"
    combined.to_csv(combined_path, index=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    summary = combined.groupby("Scenario").agg({
        "Hit": "mean",
        "First_Relevant_Rank": "mean",
    }).round(4)
    print(summary)
    print(f"\nCombined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
