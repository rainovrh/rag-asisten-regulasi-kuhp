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

_PASAL_PATTERN = re.compile(r'pasal\s+(\d+[a-zA-Z]*)', re.IGNORECASE)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_id: str
    pasal_refs: Optional[list[str]] = None
    source: str = LEGAL_CORPUS_SOURCE
    char_count: int = 0
    token_count: int = 0
    
    def __post_init__(self) -> None:
        """Calculate derived fields."""
        self.char_count = len(self.text)
        self.token_count = len(self.text.split())


class PasalSegmenter:
    """Segment legal document text by pasal boundaries."""
    
    def __init__(self) -> None:
        """Initialize pasal segmenter."""
        self._pattern = re.compile(
            r'(pasal\s+\d+[a-zA-Z]*(?:\s+ayat\s*\(\s*\d+\s*\))?)',
            re.IGNORECASE,
        )
    
    def segment(self, text: str) -> list[tuple[str, str]]:
        """Split text into pasal segments.
        
        Args:
            text: Input text.
        
        Returns:
            List of (pasal_id, pasal_text) tuples.
        """
        matches = list(self._pattern.finditer(text))
        segments = []
        
        for i, match in enumerate(matches):
            pasal_id = match.group(1).lower()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            pasal_text = text[start:end].strip()
            segments.append((pasal_id, pasal_text))
        
        logger.info(f"Segmented document into {len(segments)} pasal segments")
        return segments


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
        """Load embedding model (lazy loading)."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model
    
    def chunk_segments(
        self,
        segments: list[tuple[str, str]],
    ) -> list[Chunk]:
        """Chunk pasal segments using semantic similarity.
        
        Args:
            segments: List of (pasal_id, pasal_text) tuples.
        
        Returns:
            List of Chunk objects with pasal metadata.
        """
        logger.info(f"Starting semantic chunking with threshold {self.threshold}")
        chunks = []
        chunk_index = 0
        
        for pasal_id, pasal_text in segments:
            pasal_chunks = self._chunk_text(pasal_text, pasal_id, chunk_index)
            chunk_index += len(pasal_chunks)
            chunks.extend(pasal_chunks)
        
        logger.info(f"Created {len(chunks)} semantic chunks from {len(segments)} pasal segments")
        return chunks
    
    def _chunk_text(self, text: str, pasal_id: str, start_index: int) -> list[Chunk]:
        """Chunk a single pasal text.
        
        Args:
            text: Pasal text.
            pasal_id: Pasal identifier.
            start_index: Starting chunk index.
        
        Returns:
            List of Chunk objects.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(
                text=text,
                chunk_id=f"{pasal_id}_chunk_0",
                pasal_refs=[pasal_id],
            )]
        
        embeddings = self._compute_embeddings(sentences)
        return self._group_sentences(sentences, embeddings, pasal_id, start_index)
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        for sentence in sentences:
            parts = re.split(r'\n+', sentence)
            result.extend(part.strip() for part in parts if part.strip())
        return result
    
    def _compute_embeddings(self, sentences: list[str]) -> np.ndarray:
        """Compute embeddings for sentences."""
        model = self._load_model()
        logger.debug(f"Computing embeddings for {len(sentences)} sentences")
        return model.encode(sentences, show_progress_bar=False)
    
    def _group_sentences(
        self,
        sentences: list[str],
        embeddings: np.ndarray,
        pasal_id: str,
        start_index: int,
    ) -> list[Chunk]:
        """Group sentences into chunks based on similarity."""
        chunks = []
        current_sentences = [sentences[0]]
        chunk_index = start_index
        
        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            
            if similarity >= self.threshold:
                current_sentences.append(sentences[i])
            else:
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"{pasal_id}_chunk_{len(chunks)}",
                    pasal_refs=[pasal_id],
                ))
                current_sentences = [sentences[i]]
        
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{pasal_id}_chunk_{len(chunks)}",
                pasal_refs=[pasal_id],
            ))
        
        return chunks
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
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
    
    def chunk_segments(
        self,
        segments: list[tuple[str, str]],
    ) -> list[Chunk]:
        """Chunk pasal segments using fixed-size windows.
        
        Args:
            segments: List of (pasal_id, pasal_text) tuples.
        
        Returns:
            List of Chunk objects with pasal metadata.
        """
        logger.info("Starting fixed-size chunking")
        chunks = []
        chunk_index = 0
        
        for pasal_id, pasal_text in segments:
            pasal_chunks = self._chunk_text(pasal_text, pasal_id, chunk_index)
            chunk_index += len(pasal_chunks)
            chunks.extend(pasal_chunks)
        
        logger.info(f"Created {len(chunks)} fixed-size chunks from {len(segments)} pasal segments")
        return chunks
    
    def _chunk_text(self, text: str, pasal_id: str, start_index: int) -> list[Chunk]:
        """Chunk a single pasal text using fixed-size windows."""
        tokens = text.split()
        chunks = []
        chunk_local_index = 0
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{pasal_id}_chunk_{chunk_local_index}",
                pasal_refs=[pasal_id],
            ))
            
            chunk_local_index += 1
            start = end - self.overlap
        
        return chunks
