"""Text chunking strategies for legal documents."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.logger import get_logger
from src.config.constants import (
    DEFAULT_CHUNKING_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
    LEGAL_CORPUS_SOURCE,
)

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    pasal_ref: Optional[str] = None
    source: str = LEGAL_CORPUS_SOURCE
    char_count: int = 0
    token_count: int = 0
    
    def __post_init__(self) -> None:
        """Calculate derived fields."""
        self.char_count = len(self.text)
        self.token_count = len(self.text.split())


class SemanticChunker:
    """Semantic chunking based on sentence similarity."""
    
    def __init__(
        self,
        threshold: float = DEFAULT_CHUNKING_THRESHOLD,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """Initialize semantic chunker.
        
        Args:
            threshold: Cosine similarity threshold for grouping sentences.
            embedding_model: HuggingFace model name for embeddings.
        """
        self.threshold = threshold
        self.embedding_model_name = embedding_model
        self._model: Optional[SentenceTransformer] = None
    
    def _load_model(self) -> SentenceTransformer:
        """Load embedding model (lazy loading).
        
        Returns:
            Loaded SentenceTransformer model.
        """
        if self._model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model
    
    def chunk(self, text: str) -> list[Chunk]:
        """Split text into semantic chunks.
        
        Args:
            text: Input text to chunk.
        
        Returns:
            List of Chunk objects.
        """
        logger.info(f"Starting semantic chunking with threshold {self.threshold}")
        
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(text=text, chunk_id="chunk_0")]
        
        embeddings = self._compute_embeddings(sentences)
        chunks = self._group_sentences(sentences, embeddings)
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.
        
        Args:
            text: Input text.
        
        Returns:
            List of sentences.
        """
        # Simple sentence splitting for Indonesian
        # Split on sentence-ending punctuation followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Further split on newlines that likely indicate sentence boundaries
        result = []
        for sentence in sentences:
            parts = re.split(r'\n+', sentence)
            result.extend(part.strip() for part in parts if part.strip())
        
        return result
    
    def _compute_embeddings(self, sentences: list[str]) -> np.ndarray:
        """Compute embeddings for sentences.
        
        Args:
            sentences: List of sentences.
        
        Returns:
            Embedding matrix.
        """
        model = self._load_model()
        logger.debug(f"Computing embeddings for {len(sentences)} sentences")
        return model.encode(sentences, show_progress_bar=False)
    
    def _group_sentences(
        self,
        sentences: list[str],
        embeddings: np.ndarray,
    ) -> list[Chunk]:
        """Group sentences into chunks based on similarity.
        
        Args:
            sentences: List of sentences.
            embeddings: Sentence embeddings.
        
        Returns:
            List of Chunk objects.
        """
        chunks = []
        current_sentences = [sentences[0]]
        chunk_index = 0
        
        for i in range(1, len(sentences)):
            # Compute cosine similarity between consecutive sentences
            similarity = self._cosine_similarity(
                embeddings[i - 1], embeddings[i]
            )
            
            if similarity >= self.threshold:
                current_sentences.append(sentences[i])
            else:
                # Save current chunk and start new one
                chunk_text = " ".join(current_sentences)
                chunk_id = f"chunk_{chunk_index}"
                chunks.append(Chunk(text=chunk_text, chunk_id=chunk_id))
                chunk_index += 1
                current_sentences = [sentences[i]]
        
        # Don't forget the last chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunk_id = f"chunk_{chunk_index}"
            chunks.append(Chunk(text=chunk_text, chunk_id=chunk_id))
        
        return chunks
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            a: First vector.
            b: Second vector.
        
        Returns:
            Cosine similarity score (0-1).
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))


class FixedSizeChunker:
    """Fixed-size chunking based on token count."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> None:
        """Initialize fixed-size chunker.
        
        Args:
            chunk_size: Maximum tokens per chunk.
            overlap: Number of overlapping tokens between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> list[Chunk]:
        """Split text into fixed-size chunks.
        
        Args:
            text: Input text to chunk.
        
        Returns:
            List of Chunk objects.
        """
        tokens = text.split()
        chunks = []
        chunk_index = 0
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)
            
            chunk_id = f"chunk_{chunk_index}"
            chunks.append(Chunk(text=chunk_text, chunk_id=chunk_id))
            
            chunk_index += 1
            start = end - self.overlap
        
        logger.info(f"Created {len(chunks)} fixed-size chunks")
        return chunks
