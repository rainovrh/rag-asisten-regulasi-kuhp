"""LangChain-compatible embedding wrapper for indobenchmark/indobert-base-p1."""

from typing import Optional

import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel

from src.utils.logger import get_logger

logger = get_logger(__name__)


class IndoBERTEmbedding(Embeddings):
    """LangChain-compatible embedding wrapper using indobenchmark/indobert-base-p1 with mean pooling."""
    
    def __init__(
        self,
        model_name: str = "indobenchmark/indobert-base-p1",
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        """Initialize IndoBERT embedding model.
        
        Args:
            model_name: HuggingFace model name.
            device: Device to run model on ('cpu' or 'cuda').
            max_length: Maximum token length.
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        logger.info(f"Loading IndoBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        logger.info("IndoBERT model loaded successfully")
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.
        
        Args:
            texts: List of text strings.
        
        Returns:
            List of embedding vectors.
        """
        embeddings = []
        for text in texts:
            embedding = self._embed_text(text)
            embeddings.append(embedding.tolist())
        return embeddings
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.
        
        Args:
            text: Query text.
        
        Returns:
            Embedding vector.
        """
        return self._embed_text(text).tolist()
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Embed a single text using mean pooling.
        
        Args:
            text: Input text.
        
        Returns:
            Normalized embedding vector.
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
        
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return normalized.cpu().numpy().flatten()
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.config.hidden_size
