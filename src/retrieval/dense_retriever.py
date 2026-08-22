"""Dense retrieval using vector similarity search."""

from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.utils.logger import get_logger
from src.utils.validators import validate_file_path
from src.config.constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_DEVICE

logger = get_logger(__name__)


class DenseRetriever:
    """Dense retrieval using FAISS vector database."""
    
    def __init__(
        self,
        index_path: Optional[Path] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        device: str = DEFAULT_EMBEDDING_DEVICE,
    ) -> None:
        """Initialize dense retriever.
        
        Args:
            index_path: Path to FAISS index directory.
            embedding_model: HuggingFace model name.
            device: Device to run model on ('cpu' or 'cuda').
        
        Raises:
            FileNotFoundError: If index_path doesn't exist.
        """
        self.embedding_model = embedding_model
        self.device = device
        self._vectorstore: Optional[FAISS] = None
        
        if index_path is not None:
            self.load_index(index_path)
    
    def load_index(self, index_path: Path) -> None:
        """Load FAISS index from disk.
        
        Args:
            index_path: Path to FAISS index directory.
        
        Raises:
            FileNotFoundError: If index doesn't exist.
        """
        index_path = validate_file_path(index_path, must_exist=True)
        logger.info(f"Loading FAISS index from {index_path}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
        )
        
        self._vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded successfully")
    
    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search documents using dense retrieval.
        
        Args:
            query: Search query.
            top_k: Number of top results to return.
        
        Returns:
            List of (doc_id, score) tuples.
        
        Raises:
            RuntimeError: If index not loaded.
        """
        if self._vectorstore is None:
            raise RuntimeError("FAISS index not loaded. Call load_index() first.")
        
        results = self._vectorstore.similarity_search_with_score(query, k=top_k)
        
        formatted_results = []
        for doc, score in results:
            doc_id = doc.metadata.get("chunk_id") or doc.metadata.get("pasal", "unknown")
            formatted_results.append((doc_id, float(score)))
        
        return formatted_results
    
    def similarity_search(self, query: str, top_k: int = 10) -> list[Document]:
        """Get raw documents from similarity search.
        
        Args:
            query: Search query.
            top_k: Number of results.
        
        Returns:
            List of Document objects.
        """
        if self._vectorstore is None:
            raise RuntimeError("FAISS index not loaded. Call load_index() first.")
        
        return self._vectorstore.similarity_search(query, k=top_k)
