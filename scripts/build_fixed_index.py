#!/usr/bin/env python3
"""Build FAISS vector index for fixed-size chunks."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.paths import PROCESSED_DATA_DIR, INDEXES_DIR
from src.config.settings import settings
from src.utils.logger import setup_logger
from src.retrieval.dense_retriever import _create_embedding
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def main() -> None:
    setup_logger("INFO")
    
    # POINTING KE FIXED CORPUS
    corpus_path = PROCESSED_DATA_DIR / "kuhp_bersih_fixed.json"
    metadata_path = PROCESSED_DATA_DIR / "kuhp_bersih_fixed.metadata.json"
    index_path = INDEXES_DIR / "faiss_index_kuhp_fixed"
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)
    
    metadata_map = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_map = json.load(f)
    
    documents = []
    for doc_id, text in corpus_data.items():
        meta = metadata_map.get(doc_id, {})
        documents.append(Document(
            page_content=text,
            metadata={
                "chunk_id": doc_id,
                "source": meta.get("source", "KUHP BARU UU No 1 Tahun 2023"),
                "pasal_refs": meta.get("pasal_refs", []),
                "char_count": meta.get("char_count", len(text)),
                "token_count": meta.get("token_count", len(text.split())),
            },
        ))
    
    embedding = _create_embedding(settings.embedding_model, settings.embedding_device)
    vectorstore = FAISS.from_documents(documents, embedding)
    vectorstore.save_local(str(index_path))
    print(f"Fixed-size FAISS index saved to {index_path}")

if __name__ == "__main__":
    main()
