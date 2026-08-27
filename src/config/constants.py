"""System-wide constants."""

from pathlib import Path

# ============================================================================
# APPLICATION CONSTANTS
# ============================================================================

APP_NAME = "Asisten Hukum AI (RAG KUHP)"
APP_VERSION = "1.0.0"
DEFAULT_PAGE_TITLE = "Asisten Hukum Berbasis AI (KUHP Baru)"

# ============================================================================
# CORPUS CONSTANTS
# ============================================================================

LEGAL_CORPUS_SOURCE = "KUHP BARU UU No 1 Tahun 2023"
LEGAL_CORPUS_YEAR = 2023

# ============================================================================
# RETRIEVAL CONSTANTS
# ============================================================================

DEFAULT_SPARSE_TOP_K = 10
DEFAULT_DENSE_TOP_K = 10
DEFAULT_FINAL_TOP_K = 5
DEFAULT_RRF_K = 60
DEFAULT_CRAG_THRESHOLD = 0.010

# BM25 parameters
BM25_K1 = 1.5
BM25_B = 0.75

# ============================================================================
# CHUNKING CONSTANTS
# ============================================================================

DEFAULT_CHUNKING_THRESHOLD = 0.75
DEFAULT_CHUNKING_METHOD = "semantic"
FALLBACK_CHUNKING_METHOD = "fixed_size"

# ============================================================================
# EMBEDDING CONSTANTS
# ============================================================================

DEFAULT_EMBEDDING_MODEL = "indobenchmark/indobert-base-p1"
DEFAULT_EMBEDDING_DEVICE = "cpu"
EMBEDDING_DIMENSION = 768  # For indobenchmark/indobert-base-p1

# ============================================================================
# LLM CONSTANTS
# ============================================================================

DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_MAX_TOKENS = 64

# ============================================================================
# EVALUATION CONSTANTS
# ============================================================================

DEFAULT_GOLDEN_DATASET = "golden_dataset_rag_hukum_indonesia_rev3.csv"
DEFAULT_EVALUATION_RESULTS = "hasil_akhir_ragas_skripsi.csv"
DEFAULT_GENERATION_RESULTS = "hasil_generasi_llama3.csv"

# Evaluation metric names
METRIC_HIT_RATE = "hit_rate"
METRIC_MRR = "mrr"
METRIC_FAITHFULNESS = "faithfulness"
METRIC_ANSWER_RELEVANCE = "answer_relevancy"

# ============================================================================
# UI CONSTANTS
# ============================================================================

DEFAULT_STREAMLIT_PORT = 8501
DEFAULT_STREAMLIT_HOST = "localhost"
CHAT_INPUT_PLACEHOLDER = "Tanyakan permasalahan hukum di sini..."
PROCESSING_MESSAGE = "Mencari pasal KUHP dan Menganalisis..."

# ============================================================================
# SECURITY CONSTANTS
# ============================================================================

MIN_PASSWORD_LENGTH = 8
SESSION_TIMEOUT_MINUTES = 30
MAX_LOGIN_ATTEMPTS = 5

# ============================================================================
# LOGGING CONSTANTS
# ============================================================================

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
LOG_ROTATION = "10 MB"
LOG_RETENTION = "30 days"
LOG_ENCODING = "utf-8"
