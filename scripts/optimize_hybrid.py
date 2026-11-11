#!/usr/bin/env python3
"""Optimization and Grid-Search Script for Hybrid Retrieval Fusion (Kotak 12 & Skenario D).

Precomputes raw BM25 and FAISS Dense candidates for all 50 queries in the Golden Dataset,
then conducts an exhaustive search across:
1. Weighted RRF: w_bm25 * 1/(k + r_bm25) + w_dense * 1/(k + r_dense)
2. Dense Threshold Filtering: Only admit dense candidates if cosine similarity >= threshold
3. BM25 Rank-1 Protection / Cascading: Preserving BM25 rank 1 when confidence is high
4. Linear Score Fusion (Alpha-Fusion) with Min-Max normalization

Objective:
- Hit Rate@5 >= 85% (Baseline Skenario C: 82.00%)
- MRR >= 0.7000 (Baseline Skenario C: 0.6733)
"""

import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.paths import DATASETS_DIR, PROCESSED_DATA_DIR
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.indobert_embedding import IndoBERTEmbedding
from src.preprocessing.normalizer import LegalQueryNormalizer, UserQueryValidator


def check_hit(doc_id: str, target_chunk_id: Optional[str], target_konteks: str) -> bool:
    if not doc_id:
        return False
    if target_chunk_id and doc_id.lower() == target_chunk_id.lower():
        return True
    m_target = re.search(r"pasal\s*(\d+)", target_konteks, re.IGNORECASE)
    if not m_target:
        return False
    target_num = m_target.group(1)
    m_sem = re.search(r"kuhp_pasal_(\d+)", doc_id, re.IGNORECASE)
    if m_sem and m_sem.group(1) == target_num:
        return True
    m_fix = re.search(r"^pasal\s*(\d+)_chunk_", doc_id, re.IGNORECASE)
    if m_fix and m_fix.group(1) == target_num:
        return True
    return False


def evaluate_ranking(
    ranked_doc_ids: List[List[str]],
    targets: List[Tuple[Optional[str], str]],
) -> Dict[str, float]:
    n = len(targets)
    h1 = 0
    h3 = 0
    h5 = 0
    rrs = []

    for doc_ids, (t_cid, t_ctx) in zip(ranked_doc_ids, targets):
        hit_1 = any(check_hit(cid, t_cid, t_ctx) for cid in doc_ids[:1])
        hit_3 = any(check_hit(cid, t_cid, t_ctx) for cid in doc_ids[:3])
        hit_5 = any(check_hit(cid, t_cid, t_ctx) for cid in doc_ids[:5])
        if hit_1:
            h1 += 1
        if hit_3:
            h3 += 1
        if hit_5:
            h5 += 1

        first_rank = next((i + 1 for i, cid in enumerate(doc_ids[:5]) if check_hit(cid, t_cid, t_ctx)), None)
        rrs.append(1.0 / first_rank if first_rank else 0.0)

    return {
        "Hit@1": h1 / n,
        "Hit@3": h3 / n,
        "Hit@5": h5 / n,
        "MRR": sum(rrs) / n,
    }


def main():
    print("=" * 70)
    print("HYBRID RETRIEVAL FUSION OPTIMIZATION BENCHMARK")
    print("=" * 70)

    # 1. Load Dataset
    csv_path = DATASETS_DIR / "golden_dataset_rag_hukum_indonesia_rev3.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} golden scenarios.")

    normalizer = LegalQueryNormalizer()
    validator = UserQueryValidator()

    # 2. Load Models
    print("Loading IndoBERT & Indices...")
    emb = IndoBERTEmbedding("models/indobert-base-p1")
    bm25 = BM25Retriever.load_index(PROCESSED_DATA_DIR / "bm25_index.pkl")
    dense = DenseRetriever.load_from_dir(PROCESSED_DATA_DIR / "faiss_index_kuhp", embedding=emb)

    # 3. Precompute candidate results for all 50 queries
    print("Precomputing BM25 and Dense candidates (top 30)...")
    cached_candidates = []
    targets = []

    for _, row in df.iterrows():
        raw_q = row["Query"]
        t_cid = str(row["Target Chunk ID"]) if pd.notna(row["Target Chunk ID"]) else None
        t_ctx = str(row["Konteks Pasal"])
        targets.append((t_cid, t_ctx))

        v = validator.validate(raw_q)
        norm_q = normalizer.normalize(v.sanitized_query) or v.sanitized_query

        # BM25 candidates: list of (doc_id, score)
        bm25_res = bm25.search(norm_q, top_k=30)
        # Dense candidates: list of (doc_id, score)
        dense_res = dense.search(norm_q, top_k=30)

        cached_candidates.append({
            "query": raw_q,
            "norm_q": norm_q,
            "bm25": bm25_res,
            "dense": dense_res,
            "target": (t_cid, t_ctx),
        })

    print(f"Cached {len(cached_candidates)} query candidates successfully.\n")

    # Baseline: BM25 Alone (Semantic Chunking)
    bm25_alone = [[doc_id for doc_id, _ in c["bm25"][:5]] for c in cached_candidates]
    m_bm25 = evaluate_ranking(bm25_alone, targets)
    print(f"BASELINE: BM25 Alone (Skenario C)")
    print(f"  Hit@1: {m_bm25['Hit@1']*100:.2f}%, Hit@3: {m_bm25['Hit@3']*100:.2f}%, Hit@5: {m_bm25['Hit@5']*100:.2f}%, MRR: {m_bm25['MRR']:.4f}\n")

    # Baseline: Dense Alone
    dense_alone = [[doc_id for doc_id, _ in c["dense"][:5]] for c in cached_candidates]
    m_dense = evaluate_ranking(dense_alone, targets)
    print(f"BASELINE: Dense Alone")
    print(f"  Hit@1: {m_dense['Hit@1']*100:.2f}%, Hit@3: {m_dense['Hit@3']*100:.2f}%, Hit@5: {m_dense['Hit@5']*100:.2f}%, MRR: {m_dense['MRR']:.4f}\n")

    # Grid search results
    all_experiments = []

    # Strategy 1: Weighted RRF
    # RRF_Score(d) = w_bm25 / (k + rank_bm25) + w_dense / (k + rank_dense)
    print("--- Searching Strategy 1: Weighted RRF ---")
    for k in [10, 20, 30, 40, 50, 60]:
        for w_bm25 in [1.0, 1.2, 1.5, 2.0, 2.5]:
            for w_dense in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
                ranked = []
                for item in cached_candidates:
                    scores = {}
                    for r, (did, _) in enumerate(item["bm25"]):
                        scores[did] = scores.get(did, 0.0) + w_bm25 / (k + (r + 1))
                    for r, (did, _) in enumerate(item["dense"]):
                        scores[did] = scores.get(did, 0.0) + w_dense / (k + (r + 1))
                    sorted_dids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)[:5]
                    ranked.append(sorted_dids)

                m = evaluate_ranking(ranked, targets)
                all_experiments.append({
                    "strategy": "Weighted RRF",
                    "params": f"k={k}, w_b={w_bm25}, w_d={w_dense}",
                    **m
                })

    # Strategy 2: Weighted RRF with Dense Threshold Filtering
    print("--- Searching Strategy 2: Weighted RRF + Dense Score Thresholding ---")
    for min_dense_score in [0.60, 0.65, 0.70, 0.75, 0.80]:
        for k in [20, 30, 60]:
            for w_bm25 in [1.0, 1.5, 2.0]:
                for w_dense in [0.3, 0.5, 0.7, 1.0]:
                    ranked = []
                    for item in cached_candidates:
                        scores = {}
                        for r, (did, _) in enumerate(item["bm25"]):
                            scores[did] = scores.get(did, 0.0) + w_bm25 / (k + (r + 1))
                        for r, (did, raw_s) in enumerate(item["dense"]):
                            if raw_s >= min_dense_score:
                                scores[did] = scores.get(did, 0.0) + w_dense / (k + (r + 1))
                        sorted_dids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)[:5]
                        ranked.append(sorted_dids)

                    m = evaluate_ranking(ranked, targets)
                    all_experiments.append({
                        "strategy": "Weighted RRF + Threshold",
                        "params": f"k={k}, w_b={w_bm25}, w_d={w_dense}, min_d={min_dense_score}",
                        **m
                    })

    # Strategy 3: BM25 Rank-1 Protection / Cascading + Dense Integration
    print("--- Searching Strategy 3: BM25 Rank-1 Protection & Dense Rescue ---")
    # Idea: If BM25 has a candidate with score >= threshold or clear margin, retain it at rank 1,
    # and use weighted RRF or Dense for ranks 2..5.
    for bm25_min_margin in [1.2, 1.5, 2.0, 3.0]:
        for k in [20, 30, 60]:
            for w_dense in [0.3, 0.5, 0.7, 1.0]:
                ranked = []
                for item in cached_candidates:
                    b_res = item["bm25"]
                    d_res = item["dense"]

                    # Base RRF
                    scores = {}
                    for r, (did, _) in enumerate(b_res):
                        scores[did] = scores.get(did, 0.0) + 1.0 / (k + (r + 1))
                    for r, (did, _) in enumerate(d_res):
                        scores[did] = scores.get(did, 0.0) + w_dense / (k + (r + 1))

                    sorted_dids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)

                    # If BM25 rank 1 is significantly higher than BM25 rank 2:
                    if len(b_res) >= 2 and b_res[0][1] >= bm25_min_margin * max(b_res[1][1], 0.1):
                        b_top1 = b_res[0][0]
                        if b_top1 in sorted_dids:
                            sorted_dids.remove(b_top1)
                        sorted_dids.insert(0, b_top1)

                    ranked.append(sorted_dids[:5])

                m = evaluate_ranking(ranked, targets)
                all_experiments.append({
                    "strategy": "BM25 Protection + RRF",
                    "params": f"k={k}, w_d={w_dense}, margin={bm25_min_margin}",
                    **m
                })

    # Strategy 4: Linear Score Fusion (Alpha-Fusion) with Min-Max
    print("--- Searching Strategy 4: Min-Max Alpha-Fusion ---")
    for alpha in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        for dense_thresh in [0.0, 0.65, 0.70, 0.75]:
            ranked = []
            for item in cached_candidates:
                b_res = item["bm25"]
                d_res = item["dense"]

                # Min-max BM25
                b_max = max((s for _, s in b_res), default=1.0)
                b_min = min((s for _, s in b_res), default=0.0)
                b_range = (b_max - b_min) if (b_max - b_min) > 1e-6 else 1.0

                # Min-max Dense
                d_max = max((s for _, s in d_res), default=1.0)
                d_min = min((s for _, s in d_res), default=0.0)
                d_range = (d_max - d_min) if (d_max - d_min) > 1e-6 else 1.0

                scores = {}
                for did, raw_s in b_res:
                    norm_s = (raw_s - b_min) / b_range
                    scores[did] = scores.get(did, 0.0) + alpha * norm_s

                for did, raw_s in d_res:
                    if raw_s >= dense_thresh:
                        norm_s = (raw_s - d_min) / d_range
                        scores[did] = scores.get(did, 0.0) + (1.0 - alpha) * norm_s

                sorted_dids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)[:5]
                ranked.append(sorted_dids)

            m = evaluate_ranking(ranked, targets)
            all_experiments.append({
                "strategy": "Alpha-Fusion Min-Max",
                "params": f"alpha={alpha}, thresh={dense_thresh}",
                **m
            })

    # Filter candidates meeting our target: Hit@5 >= 0.85 and MRR >= 0.70
    df_exp = pd.DataFrame(all_experiments)
    winning = df_exp[(df_exp["Hit@5"] >= 0.85) & (df_exp["MRR"] >= 0.70)]
    winning = winning.sort_values(by=["MRR", "Hit@5", "Hit@1"], ascending=False)

    print("\n" + "=" * 70)
    print(f"TOTAL EXPERIMENTS: {len(df_exp)}")
    print(f"CONFIGURATIONS MEETING TARGET (Hit@5 >= 85% & MRR >= 0.70): {len(winning)}")
    print("=" * 70)

    if not winning.empty:
        print("\nTOP 15 BEST CONFIGURATIONS:")
        print(winning.head(15).to_string(index=False))
    else:
        print("\nTop 15 Closest Configurations by MRR:")
        print(df_exp.sort_values(by=["MRR", "Hit@5"], ascending=False).head(15).to_string(index=False))

    # Save to CSV for thesis records
    out_csv = Path("logs/comparison/hybrid_tuning_grid_search.csv")
    df_exp.to_csv(out_csv, index=False)
    print(f"\nAll grid search results saved to {out_csv}")


if __name__ == "__main__":
    main()
