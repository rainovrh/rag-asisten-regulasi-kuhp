"""IndoBERT Embedding wrapper for legal domain RAG with Mean Pooling & L2 Normalization."""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

from src.utils.logger import get_logger

logger = get_logger(__name__)

LOCAL_MODEL_PATH = Path("./models/indobert-base-p1")
DEFAULT_MODEL_NAME = "indobenchmark/indobert-base-p1"


class IndoBERTEmbedding(Embeddings):
    """LangChain-compatible embedding wrapper for IndoBERT with Mean Pooling and L2 Normalization."""

    def __init__(
        self,
        model_name_or_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        """Initialize IndoBERT embedding model.

        Args:
            model_name_or_path: Path to local model or HuggingFace hub model name.
            device: 'cpu' or 'cuda' (defaults to cuda if available).
            max_length: Maximum token length for tokenizer.
            batch_size: Batch size for document embedding to prevent OOM.
        """
        if model_name_or_path is None:
            if LOCAL_MODEL_PATH.exists():
                self.model_path = str(LOCAL_MODEL_PATH)
            else:
                self.model_path = DEFAULT_MODEL_NAME
        else:
            self.model_path = str(model_name_or_path)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.max_length = max_length
        self.batch_size = batch_size

        logger.info(f"Loading IndoBERT model from '{self.model_path}' on device '{self.device}'")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info("IndoBERT model loaded successfully with L2 Normalization enabled")

    def _mean_pooling_and_normalize(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply attention-weighted mean pooling followed by L2 Normalization."""
        # Expand attention mask to match hidden state dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # L2-Normalization: ensures ||v||_2 = 1.0 so that Inner Product == Cosine Similarity
        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return normalized

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in batches with L2 Normalization.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of 768-dimensional L2-normalized float vectors.
        """
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            encoded_inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}

            with torch.no_grad():
                model_outputs = self.model(**encoded_inputs)
                batch_vectors = self._mean_pooling_and_normalize(
                    model_outputs.last_hidden_state,
                    encoded_inputs["attention_mask"],
                )

            # Move to CPU and numpy
            numpy_vectors = batch_vectors.cpu().numpy()
            for vec in numpy_vectors:
                all_embeddings.append(vec.tolist())

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text with L2 Normalization.

        Args:
            text: Query string.

        Returns:
            768-dimensional L2-normalized float vector.
        """
        encoded_inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}

        with torch.no_grad():
            model_outputs = self.model(**encoded_inputs)
            vector = self._mean_pooling_and_normalize(
                model_outputs.last_hidden_state,
                encoded_inputs["attention_mask"],
            )

        return vector.cpu().numpy()[0].tolist()

    @property
    def dimension(self) -> int:
        """Get embedding vector dimension (768 for IndoBERT-base)."""
        return self.model.config.hidden_size
