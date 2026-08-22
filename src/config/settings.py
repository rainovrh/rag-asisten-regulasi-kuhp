"""System settings using Pydantic for validation."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application
    app_name: str = "Asisten Hukum AI (RAG KUHP)"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # Paths
    corpus_path: Path = Path("data/raw/KUHP BARU UU Nomor 1 Tahun 2023.pdf")
    processed_corpus_path: Path = Path("data/processed/kuhp_bersih.json")
    faiss_index_path: Path = Path("data/indexes/faiss_index_kuhp")
    chroma_db_path: Path = Path("data/indexes/chroma_db_kuhp")
    golden_dataset_path: Path = Path("data/datasets/golden_dataset_rag_hukum_indonesia_rev3.csv")
    log_dir: Path = Path("logs")
    
    # Embedding Model
    embedding_model: str = "indobenchmark/indobert-base-p1"
    embedding_device: str = "cpu"
    
    # LLM Configuration
    llm_model: str = "llama3"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 512
    
    # Retrieval Configuration
    sparse_top_k: int = 10
    dense_top_k: int = 10
    final_top_k: int = 5
    rrf_k: int = 60
    crag_threshold: float = 0.010
    
    # Semantic Chunking
    chunking_threshold: float = 0.75
    
    # Streamlit
    streamlit_port: int = 8501
    streamlit_host: str = "localhost"


# Global settings instance
settings = Settings()
