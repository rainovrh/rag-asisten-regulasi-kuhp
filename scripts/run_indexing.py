#!/usr/bin/env python3
"""Build FAISS vector index."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.paths import PROCESSED_DATA_DIR, INDEXES_DIR
from src.config.settings import settings
from src.utils.logger import setup_logger
from src.retrieval.dense_retriever import DenseRetriever
from src.preprocessing.chunker import SemanticChunker
import json


def main() -> None:
    """Build FAISS index."""
    setup_logger("INFO")
    
    corpus_path = PROCESSED_DATA_DIR / "kuhp_bersih.json"
    index_path = INDEXES_DIR / "faiss_index_kuhp"
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    
    # Create documents for FAISS
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    documents = []
    for doc_id, text in corpus_data.items():
        documents.append(Document(
            page_content=text,
            metadata={"chunk_id": doc_id, "source": "KUHP BARU UU No 1 Tahun 2023"},
        ))
    
    # Build index
    retriever = DenseRetriever()
    retriever._vectorstore = FAISS.from_documents(
        documents,
        HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
        ),
    )
    retriever._vectorstore.save_local(str(index_path))
    print(f"FAISS index saved to {index_path}")


if __name__ == "__main__":
    main()
