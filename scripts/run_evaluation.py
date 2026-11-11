#!/usr/bin/env python3
"""Batch Evaluation and Ablation Study Engine for Indonesian Criminal Code RAG (Blok 5, Kotak 19 & 20).

Executes the 50 Golden Dataset scenarios across 4 Ablation Scenarios:
- Skenario A: Fixed-size Chunking (500 token) + BM25 (Sparse)
- Skenario B: Fixed-size Chunking (500 token) + Hybrid (BM25 + Dense)
- Skenario C: Semantic Chunking + BM25 (Sparse)
- Skenario D: Semantic Chunking + Hybrid RRF (k=60) [Proposed Architecture]

Computes:
1. Retrieval Metrics: Hit Rate@1, Hit Rate@3, Hit Rate@5, MRR, Latency (ms).
2. Generation & RAGAS Metrics: Faithfulness, Answer Relevance, Citation Accuracy (%).
3. Exports comprehensive results to CSV for Chapter IV (Bab IV) analysis.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.paths import DATASETS_DIR, INDEXES_DIR, LOGS_DIR, PROCESSED_DATA_DIR
from src.config.settings import settings
from src.evaluation.golden_dataset import GoldenDataset
from src.evaluation.ragas_judge import RagasGroqJudge
from src.generation.guardrail import StrictLegalGroundingGuardrail
from src.generation.llm_builder import GroqLLMEngine
from src.generation.prompts import AugmentedPromptBuilder
from src.generation.response_formatter import LegalResponseFormatter
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedLegalChunk
from src.retrieval.indobert_embedding import IndoBERTEmbedding
from src.utils.logger import setup_logger

logger = setup_logger("INFO")


def check_is_hit(doc_id: str, target_chunk_id: Optional[str], target_konteks: str) -> bool:
    """Determine if a retrieved document is a true positive hit for the target article."""
    if not doc_id:
        return False

    # 1. Direct match on Target Chunk ID (Semantic Chunking)
    if target_chunk_id and doc_id.lower() == target_chunk_id.lower():
        return True

    # 2. Match based on article number across both Semantic and Fixed chunking
    m_target = re.search(r"pasal\s*(\d+)", target_konteks, re.IGNORECASE)
    if not m_target:
        return False

    target_num = m_target.group(1)

    # Check semantic ID: 'kuhp_pasal_1' or 'kuhp_pasal_1_ayat_2'
    m_sem = re.search(r"kuhp_pasal_(\d+)", doc_id, re.IGNORECASE)
    if m_sem and m_sem.group(1) == target_num:
        return True

    # Check fixed ID: 'pasal 1_chunk_0'
    m_fix = re.search(r"^pasal\s*(\d+)_chunk_", doc_id, re.IGNORECASE)
    if m_fix and m_fix.group(1) == target_num:
        return True

    return False


def evaluate_retrieval_scenario(
    scenario_name: str,
    retriever: HybridRetriever,
    dataset: GoldenDataset,
    top_k: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run retrieval evaluation for a single scenario over the 50 queries."""
    results: List[Dict[str, Any]] = []
    latencies: List[float] = []

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks: List[float] = []

    logger.info(f"--- Running Retrieval Evaluation: {scenario_name} ---")

    for s in dataset.scenarios:
        t0 = time.perf_counter()
        retrieved_chunks = retriever.search(s.query, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        retrieved_ids = [c.chunk_id for c in retrieved_chunks]

        # Calculate hits at k
        hit_1 = any(check_is_hit(cid, s.target_chunk_id, s.context) for cid in retrieved_ids[:1])
        hit_3 = any(check_is_hit(cid, s.target_chunk_id, s.context) for cid in retrieved_ids[:3])
        hit_5 = any(check_is_hit(cid, s.target_chunk_id, s.context) for cid in retrieved_ids[:5])

        if hit_1:
            hits_at_1 += 1
        if hit_3:
            hits_at_3 += 1
        if hit_5:
            hits_at_5 += 1

        # Calculate reciprocal rank
        first_rank = None
        for r_idx, cid in enumerate(retrieved_ids, start=1):
            if check_is_hit(cid, s.target_chunk_id, s.context):
                first_rank = r_idx
                break

        rr = (1.0 / first_rank) if first_rank is not None else 0.0
        reciprocal_ranks.append(rr)

        results.append({
            "Scenario": scenario_name,
            "ID": s.id,
            "Query": s.query,
            "Konteks_Pasal": s.context,
            "Target_Chunk_ID": s.target_chunk_id,
            "Ground_Truth": s.ground_truth,
            "Retrieved_Chunk_IDs": retrieved_ids,
            "Top_1_ID": retrieved_ids[0] if retrieved_ids else "",
            "Hit@1": hit_1,
            "Hit@3": hit_3,
            "Hit@5": hit_5,
            "First_Relevant_Rank": first_rank,
            "Reciprocal_Rank": rr,
            "Retrieval_Latency_ms": latency_ms,
        })

    n = len(dataset.scenarios)
    metrics = {
        "Hit_Rate@1": hits_at_1 / n if n else 0.0,
        "Hit_Rate@3": hits_at_3 / n if n else 0.0,
        "Hit_Rate@5": hits_at_5 / n if n else 0.0,
        "MRR": sum(reciprocal_ranks) / n if n else 0.0,
        "Avg_Latency_ms": sum(latencies) / n if n else 0.0,
    }

    return results, metrics


def main() -> None:
    """Execute end-to-end batch evaluation and ablation study."""
    parser = argparse.ArgumentParser(description="Run RAG Batch Evaluation & Ablation Study")
    parser.add_argument("--run-ragas", action="store_true", default=True, help="Run RAGAS evaluation via Groq API")
    parser.add_argument("--skip-ragas", action="store_true", default=False, help="Skip generation & RAGAS evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries evaluated (for fast test)")
    args = parser.parse_args()

    output_dir = LOGS_DIR / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Golden Dataset
    dataset_path = DATASETS_DIR / "golden_dataset_rag_hukum_indonesia_rev3.csv"
    dataset = GoldenDataset(dataset_path)
    if args.limit:
        dataset.scenarios = dataset.scenarios[:args.limit]
    logger.info(f"Loaded {len(dataset.scenarios)} Golden Dataset scenarios")

    # 2. Initialize Models and Indices
    logger.info("Initializing IndoBERT embedding and persistent indices...")
    emb = IndoBERTEmbedding("models/indobert-base-p1")

    # Semantic Components
    bm25_semantic = BM25Retriever.load_index(PROCESSED_DATA_DIR / "bm25_index.pkl")
    dense_semantic = DenseRetriever.load_from_dir(PROCESSED_DATA_DIR / "faiss_index_kuhp", embedding=emb)

    # Fixed-Size Components
    bm25_fixed = BM25Retriever(chunks_path=PROCESSED_DATA_DIR / "kuhp_bersih_fixed.json")
    dense_fixed = DenseRetriever.load_from_dir(INDEXES_DIR / "faiss_index_kuhp_fixed", embedding=emb)

    # 3. Configure the 4 Ablation Scenarios
    scenarios_config = [
        ("Skenario A (Baseline: Fixed BM25)", HybridRetriever(bm25_fixed, None, use_bm25=True, use_dense=False), "fixed", "bm25"),
        ("Skenario B (Fixed Hybrid BM25+Dense)", HybridRetriever(bm25_fixed, dense_fixed, use_bm25=True, use_dense=True, rrf_k=60), "fixed", "hybrid"),
        ("Skenario C (Semantic BM25)", HybridRetriever(bm25_semantic, None, use_bm25=True, use_dense=False), "semantic", "bm25"),
        ("Skenario D (Usulan: Semantic Hybrid RRF)", HybridRetriever(bm25_semantic, dense_semantic, use_bm25=True, use_dense=True, rrf_k=60), "semantic", "hybrid_rrf"),
    ]

    all_scenario_details: List[Dict[str, Any]] = []
    ablation_summary: List[Dict[str, Any]] = []

    # 4. Execute Retrieval Evaluation on all 4 Scenarios
    for name, retriever_inst, chunking, retrieval in scenarios_config:
        details, metrics = evaluate_retrieval_scenario(name, retriever_inst, dataset, top_k=5)
        all_scenario_details.extend(details)

        # Save individual scenario CSV
        safe_name = name.split("(")[0].strip().replace(" ", "_").lower()
        df_scenario = pd.DataFrame(details)
        df_scenario.to_csv(output_dir / f"{safe_name}.csv", index=False)

        ablation_summary.append({
            "Skenario": name,
            "Strategi Chunking": chunking,
            "Metode Retrieval": retrieval,
            "Hit Rate@1": f"{metrics['Hit_Rate@1'] * 100:.2f}%",
            "Hit Rate@3": f"{metrics['Hit_Rate@3'] * 100:.2f}%",
            "Hit Rate@5": f"{metrics['Hit_Rate@5'] * 100:.2f}%",
            "MRR": f"{metrics['MRR']:.4f}",
            "Avg Latency (ms)": f"{metrics['Avg_Latency_ms']:.2f} ms",
        })

    # Save summary of ablation study
    df_summary = pd.DataFrame(ablation_summary)
    summary_path = output_dir / "ringkasan_studi_ablasi_bab4.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"Ablation study summary saved to {summary_path}")

    print("\n" + "=" * 80)
    print("HASIL STUDI ABLASI RETRIEVAL (MATRIKS BAB 4 SKRIPSI):")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)

    if args.skip_ragas:
        logger.info("Skipping Generation & RAGAS evaluation as requested via --skip-ragas")
        return

    # 5. Execute Generation & RAGAS Evaluation on Proposed Architecture (Skenario D)
    llm_engine = GroqLLMEngine(model="qwen/qwen3.8-27b", temperature=0.1)
    ragas_judge = RagasGroqJudge(model="qwen/qwen3.8-27b", temperature=0.0)
    retriever_proposed = scenarios_config[3][1]

    logger.info("--- Executing Generation & RAGAS Evaluation on Skenario D (Proposed Architecture) ---")
    gen_results: List[Dict[str, Any]] = []
    full_eval_path = output_dir / "evaluasi_lengkap_skripsi_bab4.csv"

    # Resume from existing checkpoint if present
    completed_ids = set()
    if full_eval_path.exists():
        try:
            existing_df = pd.read_csv(full_eval_path)
            if not existing_df.empty and "ID" in existing_df.columns:
                gen_results = existing_df.to_dict(orient="records")
                completed_ids = set(existing_df["ID"].tolist())
                logger.info(f"Resuming evaluation: found {len(completed_ids)} already completed scenarios")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")

    faithfulness_scores: List[float] = [r["RAGAS_Faithfulness"] for r in gen_results if "RAGAS_Faithfulness" in r and pd.notna(r["RAGAS_Faithfulness"])]
    relevance_scores: List[float] = [r["RAGAS_Answer_Relevance"] for r in gen_results if "RAGAS_Answer_Relevance" in r and pd.notna(r["RAGAS_Answer_Relevance"])]
    citation_accuracies: List[float] = [r["Citation_Accuracy"] for r in gen_results if "Citation_Accuracy" in r and pd.notna(r["Citation_Accuracy"])]

    for s in tqdm(dataset.scenarios, desc="Evaluating Generation & RAGAS"):
        if s.id in completed_ids:
            continue

        # Retrieve chunks with proposed retriever
        chunks = retriever_proposed.search(s.query, top_k=5)

        # Generate answer through Blok 4 pipeline
        legal_ans = llm_engine.generate_legal_answer(s.query, chunks)

        # Combine retrieved context text for judge
        combined_context = "\n\n".join([f"[{c.pasal}]\n{c.text}" for c in chunks])

        # Evaluate via RAGAS Judge
        ragas_eval = ragas_judge.evaluate_single(
            query=s.query,
            context=combined_context,
            answer=legal_ans.full_markdown,
        )

        f_score = ragas_eval["faithfulness"]
        r_score = ragas_eval["answer_relevance"]
        c_acc = legal_ans.citation_accuracy

        faithfulness_scores.append(f_score)
        relevance_scores.append(r_score)
        citation_accuracies.append(c_acc)

        gen_results.append({
            "ID": s.id,
            "Query": s.query,
            "Konteks_Pasal": s.context,
            "Target_Chunk_ID": s.target_chunk_id,
            "Ground_Truth": s.ground_truth,
            "AI_Kesimpulan": legal_ans.kesimpulan,
            "AI_Dasar_Hukum": legal_ans.dasar_hukum,
            "Grounding_Status": legal_ans.grounding_status,
            "Citation_Accuracy": c_acc,
            "Sitasi_Valid": legal_ans.sitasi_valid,
            "RAGAS_Faithfulness": f_score,
            "RAGAS_Answer_Relevance": r_score,
            "Reasoning_Faithfulness": ragas_eval.get("reasoning_faithfulness", ""),
            "Reasoning_Relevance": ragas_eval.get("reasoning_relevance", ""),
            "Inference_Time_s": legal_ans.inference_time_seconds,
            "AI_Full_Markdown": legal_ans.full_markdown,
        })

        # Incremental checkpointing after each query
        full_eval_path = output_dir / "evaluasi_lengkap_skripsi_bab4.csv"
        pd.DataFrame(gen_results).to_csv(full_eval_path, index=False)

    # 6. Save Full Results
    df_gen = pd.DataFrame(gen_results)
    logger.info(f"Full generation & RAGAS results saved to {full_eval_path}")

    # 7. Print Comprehensive Summary
    avg_f = float(df_gen["RAGAS_Faithfulness"].mean())
    avg_r = float(df_gen["RAGAS_Answer_Relevance"].mean())
    avg_c = float(df_gen["Citation_Accuracy"].mean())

    print("\n" + "=" * 80)
    print("HASIL STUDI ABLASI RETRIEVAL (MATRIKS BAB 4 SKRIPSI):")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)

    print("\nHASIL EVALUASI GENERATION & RAGAS (SKENARIO D - USULAN UTAMA):")
    print(f"- Rata-rata Faithfulness       : {avg_f:.4f} ({avg_f * 100:.2f}%) [Target >= 85%: {'PASSED' if avg_f >= 0.85 else 'BELOW TARGET'}]")
    print(f"- Rata-rata Answer Relevance   : {avg_r:.4f} ({avg_r * 100:.2f}%) [Target >= 80%: {'PASSED' if avg_r >= 0.80 else 'BELOW TARGET'}]")
    print(f"- Rata-rata Citation Accuracy  : {avg_c:.4f} ({avg_c * 100:.2f}%) [Target 100%: {'PASSED' if avg_c == 1.0 else 'CHECK DISCREPANCY'}]")
    print("=" * 80)


if __name__ == "__main__":
    main()
