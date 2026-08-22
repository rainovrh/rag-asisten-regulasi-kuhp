"""Evaluation runner orchestrator."""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.utils.logger import get_logger
from src.config.settings import settings
from src.evaluation.golden_dataset import GoldenDataset
from src.evaluation.metrics import calculate_hit_rate, calculate_mrr
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm import LLMEngine
from src.generation.prompts import get_legal_qa_prompt

logger = get_logger(__name__)

_PASAL_PATTERN = re.compile(r'(pasal\s+\d+[a-zA-Z]*(?:\s+ayat\s*\(\s*\d+\s*\))?)', re.IGNORECASE)


class EvaluationRunner:
    """Orchestrate evaluation runs."""
    
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMEngine,
        golden_dataset: GoldenDataset,
        pasal_mapping_path: Optional[Path] = None,
    ) -> None:
        """Initialize evaluation runner.
        
        Args:
            retriever: Hybrid retriever instance.
            llm: LLM engine instance.
            golden_dataset: Golden dataset instance.
            pasal_mapping_path: Optional path to pasal-to-chunks mapping JSON.
        """
        self.retriever = retriever
        self.llm = llm
        self.dataset = golden_dataset
        self._pasal_mapping: dict[str, list[str]] = {}
        
        if pasal_mapping_path is not None and pasal_mapping_path.exists():
            with open(pasal_mapping_path, "r", encoding="utf-8") as f:
                self._pasal_mapping = json.load(f)
            logger.info(f"Loaded pasal mapping with {len(self._pasal_mapping)} entries")
        else:
            logger.warning("No pasal mapping provided; hit detection will use exact match only")
    
    def _get_pasal_chunks(self, pasal_ref: str) -> set[str]:
        """Get chunk IDs associated with a pasal reference.
        
        Args:
            pasal_ref: Pasal reference string (e.g., "Pasal 1 ayat (1)").
        
        Returns:
            Set of chunk IDs.
        """
        match = _PASAL_PATTERN.search(pasal_ref)
        if not match:
            return set()
        
        pasal_key = match.group(1).lower()
        return set(self._pasal_mapping.get(pasal_key, []))
    
    def run(self, output_path: Optional[Path] = None, run_ragas: bool = False) -> pd.DataFrame:
        """Run evaluation on all scenarios.
        
        Args:
            output_path: Path to save results CSV.
            run_ragas: Whether to run RAGAS evaluation (Faithfulness, Answer Relevance).
        
        Returns:
            DataFrame with evaluation results.
        """
        logger.info("Starting evaluation run")
        
        queries = self.dataset.get_queries()
        ground_truths = self.dataset.get_ground_truths()
        contexts = self.dataset.get_contexts()
        
        results = []
        
        for i, (query, gt, ctx) in enumerate(tqdm(
            zip(queries, ground_truths, contexts),
            total=len(queries),
            desc="Evaluating",
        )):
            try:
                # Retrieve documents
                retrieved = self.retriever.search(query, top_k=5, apply_crag=True)
                retrieved_doc_ids = [doc_id for doc_id, _, _ in retrieved]
                
                # Check hit using pasal reference (ctx), not answer text (gt)
                expected_chunks = self._get_pasal_chunks(ctx)
                if expected_chunks:
                    hit = any(doc_id in expected_chunks for doc_id in retrieved_doc_ids)
                else:
                    hit = ctx in retrieved_doc_ids
                
                # Find rank of first relevant document
                first_relevant_rank = None
                for rank, (doc_id, _, _) in enumerate(retrieved, start=1):
                    if expected_chunks:
                        if doc_id in expected_chunks:
                            first_relevant_rank = rank
                            break
                    else:
                        if doc_id == ctx:
                            first_relevant_rank = rank
                            break
                
                # Generate answer
                if retrieved:
                    context_text = "\n\n".join(
                        f"[{doc_id}]: {text}"
                        for doc_id, _, text in retrieved
                    )
                    prompt = get_legal_qa_prompt()
                    answer = self.llm.generate_with_template(prompt, context_text, query)
                else:
                    answer = "Informasi tidak tersedia di konteks. (Dicegah oleh filter CRAG)"
                
                results.append({
                    "ID": i + 1,
                    "Query": query,
                    "Konteks_Pasal": ctx,
                    "Ground_Truth": gt,
                    "Retrieved_Context": "\n\n".join(
                        f"[{doc_id}]: {text}" for doc_id, _, text in retrieved
                    ) if retrieved else "",
                    "AI_Answer": answer,
                    "Hit": hit,
                    "First_Relevant_Rank": first_relevant_rank,
                })
                
            except Exception as e:
                logger.error(f"Error evaluating query {i+1}: {e}")
                results.append({
                    "ID": i + 1,
                    "Query": query,
                    "Konteks_Pasal": ctx,
                    "Ground_Truth": gt,
                    "Retrieved_Context": "",
                    "AI_Answer": f"Error: {str(e)}",
                    "Hit": False,
                    "First_Relevant_Rank": None,
                })
        
        df = pd.DataFrame(results)
        
        # Calculate aggregate metrics
        hit_rate = calculate_hit_rate(
            [[(doc_id, score) for doc_id, score, _ in r]
             for r in results],
            contexts,
            top_k=5,
            pasal_mapping=self._pasal_mapping or None,
        )
        mrr = calculate_mrr(
            [[(doc_id, score) for doc_id, score, _ in r]
             for r in results],
            contexts,
            pasal_mapping=self._pasal_mapping or None,
        )
        
        logger.info(f"Hit Rate@5: {hit_rate:.4f}")
        logger.info(f"MRR: {mrr:.4f}")
        
        # RAGAS evaluation
        if run_ragas:
            df = self._run_ragas(df)
        
        # Save results
        if output_path is None:
            output_path = settings.log_dir / "evaluation_results.csv"
        
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        json_output_path = output_path.with_suffix(".json")
        df.to_json(json_output_path, orient="records", indent=2, force_ascii=False)
        logger.info(f"Results saved to {json_output_path}")
        
        return df
    
    def _run_ragas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run RAGAS evaluation on generated answers.
        
        Args:
            df: DataFrame with evaluation results.
        
        Returns:
            DataFrame with RAGAS scores added.
        """
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy
            from datasets import Dataset
        except ImportError:
            logger.warning("RAGAS not installed, skipping RAGAS evaluation")
            df["faithfulness"] = None
            df["answer_relevancy"] = None
            return df
        
        logger.info("Running RAGAS evaluation...")
        
        # Prepare data for RAGAS
        questions = df["Query"].astype(str).tolist()
        answers = df["AI_Answer"].astype(str).tolist()
        
        contexts = []
        for ctx in df["Retrieved_Context"]:
            if pd.isna(ctx) or not str(ctx).strip():
                contexts.append([""])
            else:
                contexts.append([str(ctx)])
        
        data_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        ragas_dataset = Dataset.from_dict(data_dict)
        
        # Run evaluation
        result = evaluate(
            dataset=ragas_dataset,
            metrics=[faithfulness, answer_relevancy],
        )
        
        result_df = result.to_pandas()
        
        # Merge with original df
        df = pd.concat([df, result_df[["faithfulness", "answer_relevancy"]]], axis=1)
        
        logger.info(f"Avg Faithfulness: {df['faithfulness'].mean():.4f}")
        logger.info(f"Avg Answer Relevancy: {df['answer_relevancy'].mean():.4f}")
        
        return df
