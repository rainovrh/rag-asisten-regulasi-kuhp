#!/usr/bin/env python3
"""Run comparison only for Baseline vs Proposed to save API quota."""

import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.paths import LOGS_DIR, PROCESSED_DATA_DIR, INDEXES_DIR, DATASETS_DIR
from src.config.settings import settings
from src.utils.logger import setup_logger
from src.evaluation.golden_dataset import GoldenDataset
from src.evaluation.runner import EvaluationRunner
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.llm import LLMEngine

def run_scenario(name, chunking, retrieval, corpus_path, index_path, dataset, pasal_mapping_path, output_dir, limit=None, provider="groq", model=None):
    print(f"\n--- Running Scenario: {name} (provider: {provider}) ---")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    
    bm25 = BM25Retriever(corpus_data=corpus_data)
    dense = DenseRetriever(index_path=index_path)
    retriever = HybridRetriever(bm25, dense, use_bm25=(retrieval in ["bm25", "hybrid"]), use_dense=(retrieval in ["dense", "hybrid"]))
    
    kwargs = {"provider": provider}
    if model:
        kwargs["model"] = model
    llm = LLMEngine(**kwargs)
    
    runner = EvaluationRunner(retriever, llm, dataset, pasal_mapping_path=pasal_mapping_path)
    
    df = runner.run(output_dir / f"{name}_results.csv", limit=limit)
    df["Scenario"] = name
    return df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Number of queries to evaluate")
    parser.add_argument("--provider", type=str, default="groq", choices=["groq", "ollama"], help="LLM provider")
    parser.add_argument("--model", type=str, default=None, help="Model name override")
    args = parser.parse_args()
    
    setup_logger("INFO")
    output_dir = LOGS_DIR / "core_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = GoldenDataset(settings.golden_dataset_path)
    pasal_mapping_path = PROCESSED_DATA_DIR / "pasal_to_chunks.json"
    
    results = []
    # Baseline
    results.append(run_scenario("A_Baseline", "fixed", "bm25", PROCESSED_DATA_DIR/"kuhp_bersih_fixed.json", INDEXES_DIR/"faiss_index_kuhp_fixed", dataset, pasal_mapping_path, output_dir, limit=args.limit, provider=args.provider, model=args.model))
    # Proposed
    results.append(run_scenario("D_Usulan", "semantic", "hybrid", PROCESSED_DATA_DIR/"kuhp_bersih.json", INDEXES_DIR/"faiss_index_kuhp", dataset, pasal_mapping_path, output_dir, limit=args.limit, provider=args.provider, model=args.model))
    
    combined = pd.concat(results, ignore_index=True)
    combined.to_csv(output_dir / "core_results.csv", index=False)
    print("\nSummary:")
    print(combined.groupby("Scenario")[["Hit", "First_Relevant_Rank"]].mean())
