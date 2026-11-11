"""Dense retrieval using FAISS Vector Database with IndoBERT and Cosine Similarity."""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from src.config.constants import DEFAULT_DENSE_TOP_K
from src.retrieval.indobert_embedding import IndoBERTEmbedding
from src.utils.logger import get_logger
from src.utils.validators import validate_file_path

logger = get_logger(__name__)


class DenseRetriever:
    """Dense semantic retrieval using FAISS IndexFlatIP and L2-Normalized IndoBERT."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        embedding: Optional[IndoBERTEmbedding] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize dense retriever.

        Args:
            index_path: Path to serialized FAISS index directory.
            embedding: Pre-initialized IndoBERTEmbedding instance.
            device: 'cpu' or 'cuda'.
        """
        self.device = device
        self._embedding = embedding or IndoBERTEmbedding(device=device)
        self._vectorstore: Optional[FAISS] = None

        if index_path is not None:
            self.load_index(index_path)

    def build_from_chunks(
        self,
        chunks: List[dict],
        batch_size: int = 32,
    ) -> FAISS:
        """Build FAISS IndexFlatIP from list of chunk dictionaries.

        Args:
            chunks: List of chunk dictionaries containing 'id_unik', 'isi_teks', 'metadata'.
            batch_size: Batch size for embedding computation.

        Returns:
            Configured FAISS vectorstore.
        """
        logger.info(f"Building FAISS IndexFlatIP for {len(chunks)} legal chunks (batch_size={batch_size})")

        documents: List[Document] = []
        for c in chunks:
            chunk_id = c.get("id_unik", "")
            text = c.get("isi_teks", "")
            meta = c.get("metadata", {})
            meta_clean = dict(meta)
            meta_clean["chunk_id"] = chunk_id

            doc = Document(
                page_content=text,
                metadata=meta_clean,
            )
            documents.append(doc)

        # Build FAISS with DistanceStrategy.COSINE (IndexFlatIP + L2 normalization)
        self._vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self._embedding,
            distance_strategy=DistanceStrategy.COSINE,
        )
        logger.info(f"FAISS IndexFlatIP successfully built with {self._vectorstore.index.ntotal} vectors")
        return self._vectorstore

    def save_index(self, output_dir: Path) -> None:
        """Persist FAISS index and metadata to disk."""
        if self._vectorstore is None:
            raise RuntimeError("Cannot save index: vectorstore is not built or loaded")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save_local(str(output_dir))
        logger.info(f"Saved FAISS index ({self._vectorstore.index.ntotal} vectors) to {output_dir}")

    def load_index(self, index_dir: Path) -> None:
        """Load persistent FAISS index from disk."""
        index_dir = validate_file_path(index_dir, must_exist=True)
        logger.info(f"Loading persistent FAISS index from {index_dir}")
        self._vectorstore = FAISS.load_local(
            str(index_dir),
            self._embedding,
            distance_strategy=DistanceStrategy.COSINE,
            allow_dangerous_deserialization=True,
        )
        self._doc_id_map: dict[str, Document] = {}
        if hasattr(self._vectorstore, "docstore") and hasattr(self._vectorstore.docstore, "_dict"):
            for doc in self._vectorstore.docstore._dict.values():
                cid = doc.metadata.get("chunk_id", doc.metadata.get("id_unik", ""))
                if cid:
                    self._doc_id_map[cid] = doc
        logger.info(f"Loaded FAISS index with {self._vectorstore.index.ntotal} vectors (metric={self._vectorstore.index.metric_type})")

    @classmethod
    def load_from_dir(
        cls,
        index_dir: Path,
        embedding: Optional[IndoBERTEmbedding] = None,
        device: Optional[str] = None,
    ) -> "DenseRetriever":
        """Instantiate and load dense retriever from directory."""
        retriever = cls(embedding=embedding, device=device)
        retriever.load_index(index_dir)
        return retriever

    def get_document(self, doc_id: str) -> str:
        """Get document text by unique chunk ID."""
        if hasattr(self, "_doc_id_map") and doc_id in self._doc_id_map:
            return self._doc_id_map[doc_id].page_content
        raise KeyError(f"Chunk ID '{doc_id}' not found in FAISS docstore")

    def get_metadata(self, doc_id: str) -> dict:
        """Get document metadata by unique chunk ID."""
        if hasattr(self, "_doc_id_map") and doc_id in self._doc_id_map:
            return self._doc_id_map[doc_id].metadata
        raise KeyError(f"Chunk ID '{doc_id}' not found in FAISS docstore")

    def search(self, query: str, top_k: int = DEFAULT_DENSE_TOP_K) -> List[Tuple[str, float]]:
        """Perform semantic similarity search on legal chunks.

        Args:
            query: User inquiry or legal case scenario.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of (chunk_id, similarity_score) tuples sorted descending by relevance.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty")

        if self._vectorstore is None:
            raise RuntimeError("FAISS vectorstore is not loaded. Call load_index() or build_from_chunks() first.")

        # similarity_search_with_score returns (Document, cosine_similarity_score)
        # without triggering spurious range warnings on negative cosine values
        results = self._vectorstore.similarity_search_with_score(query, k=top_k)

        scored_results: List[Tuple[str, float]] = []
        for doc, score in results:
            chunk_id = doc.metadata.get("chunk_id", doc.metadata.get("id_unik", ""))
            scored_results.append((chunk_id, float(score)))

        return scored_results

    @property
    def total_vectors(self) -> int:
        """Number of vectors currently stored in the FAISS index."""
        if self._vectorstore is None or self._vectorstore.index is None:
            return 0
        return self._vectorstore.index.ntotal
