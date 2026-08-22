# STYLE GUIDE
## Sistem Asisten Regulasi RAG untuk Domain Hukum Indonesia

---

## DOKUMEN PENGENDALI VERSI

| Versi | Tanggal | Penulis | Deskripsi |
|-------|---------|---------|-----------|
| 1.0 | 2026-08-13 | Rainova Rahaniawan (152023007) | Pedoman coding standards dan arsitektur proyek |

---

## DAFTAR ISI

1. [Prinsip Umum](#1-prinsip-umum)
2. [Struktur Direktori](#2-struktur-direktori)
3. [Konvensi Penamaan](#3-konvensi-penamaan)
4. [Style Guide Python](#4-style-guide-python)
5. [Type Hints & Static Analysis](#5-type-hints--static-analysis)
6. [Documentation Standards](#6-documentation-standards)
7. [Error Handling & Logging](#7-error-handling--logging)
8. [Testing Standards](#8-testing-standards)
9. [Git Workflow & Commit Convention](#9-git-workflow--commit-convention)
10. [Dependency Management](#10-dependency-management)
11. [Security Standards](#11-security-standards)
12. [Performance Guidelines](#12-performance-guidelines)

---

## 1. PRINSIP UMUM

### 1.1 Filosofi Arsitektur

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ARSITEKTUR SISTEM                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Separation of Concerns                                           │
│     - Setiap modul memiliki tanggung jawab tunggal                    │
│     - Tidak ada god-class atau spaghetti code                         │
│                                                                     │
│  2. Dependency Inversion                                             │
│     - Modul tingkat tinggi tidak bergantung pada modul tingkat rendah │
│     - Abstraksi melalui interface/protocol                           │
│                                                                     │
│  3. Explicit over Implicit                                           │
│     - Kode harus mudah dibaca dan dipahami                           │
│     - Hindari magic numbers, magic strings, implicit behavior       │
│                                                                     │
│  4. Fail Fast & Fail Safe                                           │
│     - Validasi input di awal                                        │
│     - Graceful degradation untuk komponen opsional                  │
│                                                                     │
│  5. Testability                                                     │
│     - Setiap fungsi dapat diuji secara mandiri                      │
│     - Dependency injection untuk mocking                            │
│                                                                     │
│  6. Reproducibility                                                 │
│     - Semua eksperimen dapat direproduksi                            │
│     - Version pinning untuk semua dependensi                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Code Quality Principles

| Prinsip | Deskripsi | Contoh |
|---------|-----------|--------|
| **DRY** | Don't Repeat Yourself | Fungsi retrieval tidak di-duplikasi di fase3 dan fase4 |
| **KISS** | Keep It Simple, Stupid | Gunakan tipe data sederhana yang cukup |
| **YAGNI** | You Aren't Gonna Need It | Jangan tambah fitur yang tidak dibutuhkan untuk thesis |
| **SOLID** | Single Responsibility, Open/Closed, Liskov, Interface Segregation, Dependency Inversion | Setiap class/fungsi melakukan satu hal |
| **PEP 20** | The Zen of Python | Readability counts, Simple is better than complex |

---

## 2. STRUKTUR DIREKTORI

### 2.1 Directory Tree

```
File Hukum/
├── src/                                    # Source code utama
│   ├── __init__.py                         # Package marker
│   ├── config/                             # Konfigurasi terpusat
│   │   ├── __init__.py
│   │   ├── settings.py                     # Settings dengan Pydantic
│   │   ├── paths.py                        # Path management
│   │   └── constants.py                    # Konstanta sistem
│   ├── preprocessing/                      # Pipeline pra-pemrosesan
│   │   ├── __init__.py
│   │   ├── cleaner.py                      # PDF cleaning & extraction
│   │   ├── normalizer.py                   # Text normalization
│   │   └── chunker.py                      # Semantic & fixed-size chunking
│   ├── retrieval/                          # Pipeline retrieval
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py               # Sparse retrieval (BM25)
│   │   ├── dense_retriever.py              # Dense retrieval (FAISS)
│   │   ├── hybrid_retriever.py             # Hybrid + RRF
│   │   └── reranker.py                     # CRAG & filtering
│   ├── generation/                         # Pipeline generasi
│   │   ├── __init__.py
│   │   ├── llm.py                          # LLM interface (Ollama)
│   │   └── prompts.py                      # Prompt templates
│   ├── evaluation/                         # Pipeline evaluasi
│   │   ├── __init__.py
│   │   ├── metrics.py                      # Hit Rate, MRR, RAGAS
│   │   ├── golden_dataset.py               # Dataset management
│   │   └── runner.py                       # Evaluation orchestrator
│   ├── ui/                                 # Antarmuka pengguna
│   │   ├── __init__.py
│   │   ├── app.py                          # Streamlit main app
│   │   ├── components.py                   # Reusable UI components
│   │   └── styles.py                       # Custom CSS
│   └── utils/                              # Utilities
│       ├── __init__.py
│       ├── logger.py                       # Logging configuration
│       ├── validators.py                   # Input validation
│       └── helpers.py                      # Helper functions
├── tests/                                  # Unit & integration tests
│   ├── __init__.py
│   ├── conftest.py                         # Pytest fixtures
│   ├── test_preprocessing.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_evaluation.py
├── data/                                   # Data files
│   ├── raw/                                # Original documents
│   │   └── KUHP BARU UU Nomor 1 Tahun 2023.pdf
│   ├── processed/                          # Processed documents
│   │   ├── kuhp_bersih.json
│   │   └── chunks/
│   ├── indexes/                            # Vector databases
│   │   ├── faiss_index_kuhp/
│   │   └── chroma_db_kuhp/
│   └── datasets/                           # Evaluation datasets
│       ├── golden_dataset_rag_hukum_indonesia_rev3.csv
│       ├── hasil_generasi_llama3.csv
│       └── hasil_akhir_ragas_skripsi.csv
├── scripts/                                # Entry point scripts
│   ├── run_preprocessing.py
│   ├── run_indexing.py
│   ├── run_evaluation.py
│   └── run_app.py
├── docs/                                   # Documentation
│   ├── SPECIFICATION.md
│   ├── TECH_STACK.md
│   ├── ASSESSMENT.md
│   ├── database.sql
│   ├── STYLE_GUIDE.md
│   └── CONTRIBUTING.md
├── logs/                                   # Application logs
├── venv/                                   # Virtual environment (gitignored)
├── .gitignore
├── .env.example
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── CONTRIBUTING.md
└── README.md
```

### 2.2 Directory Rules

| Directory | Rule | Justifikasi |
|-----------|------|-------------|
| `src/` | Semua kode Python production | Pemisahan jelas antara kode dan data |
| `tests/` | Mirror struktur `src/` | Test discovery otomatis oleh pytest |
| `data/` | Data di luar version control (kecuali raw) | Dataset besar tidak masuk git |
| `scripts/` | Entry points yang dapat dieksekusi | Single responsibility untuk CLI |
| `docs/` | Semua dokumentasi proyek | Single source of truth |
| `logs/` | Log files (gitignored) | Sensitive, large, ephemeral |
| `venv/` | Virtual environment (gitignored) | Standard Python practice |

---

## 3. KONVENSI PENAMAAN

### 3.1 General Naming Rules

| Konteks | Format | Contoh |
|---------|--------|--------|
| **File Python** | `snake_case.py` | `hybrid_retriever.py` |
| **Direktori** | `snake_case/` | `retrieval/` |
| **Class** | `PascalCase` | `HybridRetriever`, `BM25Retriever` |
| **Function/Method** | `snake_case()` | `retrieve_context()`, `calculate_mrr()` |
| **Variable** | `snake_case` | `query_text`, `bm25_scores` |
| **Constant** | `UPPER_SNAKE_CASE` | `DEFAULT_TOP_K`, `RRF_K_CONSTANT` |
| **Private** | `_leading_underscore` | `_internal_method()` |
| **Type Alias** | `PascalCase` | `DocumentId`, `RetrievalResult` |

### 3.2 Domain-Specific Naming

| Konsep | Nama yang Disukai | Nama yang Ditolak |
|--------|-------------------|-------------------|
| Legal document chunk | `pasal_chunk` | `legal_text`, `document_piece` |
| Query from user | `user_query` | `input`, `question_text` |
| Retrieved context | `retrieved_context` | `result`, `output` |
| Ground truth | `ground_truth` | `answer`, `correct_answer` |
| Evaluation run | `evaluation_run` | `test`, `experiment` |
| RRF score | `rrf_score` | `rank_score`, `fusion_score` |

### 3.3 File Naming by Module

| Module | File Pattern | Contoh |
|--------|--------------|--------|
| Preprocessing | `{action}_*.py` | `cleaner.py`, `normalizer.py`, `chunker.py` |
| Retrieval | `{method}_retriever.py` | `bm25_retriever.py`, `dense_retriever.py` |
| Generation | `{component}.py` | `llm.py`, `prompts.py` |
| Evaluation | `{metric_or_action}.py` | `metrics.py`, `runner.py` |
| UI | `{component}.py` | `app.py`, `components.py` |

---

## 4. STYLE GUIDE PYTHON

### 4.1 Formatting Standards

**Tool: Black** (line length: 100)

```python
# GOOD - Formatted by Black
def calculate_rrf_score(
    bm25_ranks: list[str],
    faiss_ranks: list[str],
    k: int = 60,
) -> dict[str, float]:
    ...

# BAD - Inconsistent formatting
def calculate_rrf_score(bm25_ranks,faiss_ranks,k=60):
    ...
```

**Tool: isort** (import sorting)

```python
# GOOD - Standard library → Third-party → Local
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS

from src.config.settings import settings
from src.retrieval.bm25_retriever import BM25Retriever
```

### 4.2 Line Length & Structure

| Aturan | Nilai | Keterangan |
|--------|-------|------------|
| **Max Line Length** | 100 characters | Black default |
| **Max Function Length** | 50 lines | Jika lebih, pecah menjadi fungsi kecil |
| **Max File Length** | 500 lines | Jika lebih, pisah menjadi module |
| **Max Parameter Count** | 5 parameters | Gunakan dataclass/dict untuk lebih |
| **Max Nesting Depth** | 4 levels | Gunakan early return untuk flatten |

### 4.3 Function Design

```python
# GOOD - Single responsibility, clear signature
def retrieve_legal_documents(
    query: str,
    top_k: int = 10,
    similarity_threshold: float = 0.75,
) -> list[RetrievalResult]:
    """Retrieve legal documents using hybrid search.
    
    Args:
        query: User's legal question in Indonesian.
        top_k: Maximum number of documents to retrieve.
        similarity_threshold: Minimum similarity score (0-1).
    
    Returns:
        List of retrieved documents with scores.
    
    Raises:
        ValueError: If query is empty or top_k is negative.
        ConnectionError: If vector database is unavailable.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    
    if top_k < 1:
        raise ValueError("top_k must be positive")
    
    # Implementation here
    ...

# BAD - Multiple responsibilities, unclear signature
def do_stuff(q, k=10, t=0.75):
    results = []
    # ... 100 lines of mixed logic
    return results
```

### 4.4 Class Design

```python
# GOOD - Clean class with clear responsibilities
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RetrievalResult:
    """Single retrieval result with metadata."""
    doc_id: str
    text: str
    score: float
    source: str
    pasal_ref: Optional[str] = None

class HybridRetriever:
    """Hybrid retrieval combining BM25 and dense search."""
    
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
    ) -> None:
        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        self._rrf_k = rrf_k
    
    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Execute hybrid search with RRF fusion."""
        bm25_results = self._bm25.search(query, top_k)
        dense_results = self._dense.search(query, top_k)
        return self._fuse_results(bm25_results, dense_results, top_k)
    
    def _fuse_results(
        self,
        bm25_results: list[RetrievalResult],
        dense_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Fuse results using Reciprocal Rank Fusion."""
        # Private implementation
        ...

# BAD - God class
class System:
    def __init__(self):
        self.data = None
        self.model = None
        self.results = []
    
    def do_everything(self):
        # 500 lines of mixed concerns
        pass
```

### 4.5 Docstring Standards (Google Style)

```python
def semantic_chunking(
    text: str,
    threshold: float = 0.75,
    embedding_model: str = "indobenchmark/indobert-base-p1",
) -> list[str]:
    """Split text into semantic chunks based on sentence similarity.
    
    This function implements semantic chunking by:
    1. Splitting text into sentences
    2. Computing embeddings for each sentence
    3. Grouping sentences with cosine similarity above threshold
    
    Args:
        text: Input text to be chunked (already cleaned).
        threshold: Minimum cosine similarity to group sentences (0-1).
        embedding_model: HuggingFace model name for embeddings.
    
    Returns:
        List of text chunks, each containing semantically related sentences.
    
    Raises:
        ValueError: If text is empty or threshold is out of range.
        RuntimeError: If embedding model fails to load.
    
    Example:
        >>> text = "Pasal 1. Setiap orang bebas... Pasal 2. Hak asasi..."
        >>> chunks = semantic_chunking(text, threshold=0.75)
        >>> len(chunks)
        2
    """
    if not text.strip():
        raise ValueError("Input text cannot be empty")
    
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    
    # Implementation
    ...
```

### 4.6 Import Organization

```
# 1. Standard library
import json
import logging
import re
from pathlib import Path
from typing import Optional, Union

# 2. Third-party libraries
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 3. Local application imports
from src.config.settings import settings
from src.preprocessing.chunker import semantic_chunking
from src.retrieval.bm25_retriever import BM25Retriever
```

**Absolute imports only** - Never use relative imports within `src/`.

---

## 5. TYPE HINTS & STATIC ANALYSIS

### 5.1 Type Hints Requirements

```python
# REQUIRED - All function signatures MUST have type hints
from typing import Optional

def load_corpus(path: Path, encoding: str = "utf-8") -> dict[str, str]:
    """Load legal corpus from JSON file."""
    with open(path, encoding=encoding) as f:
        return json.load(f)

# Use typing_extensions for older Python compatibility
from typing_extensions import TypeAlias

DocumentId: TypeAlias = str
Score: TypeAlias = float

# Use dataclasses for complex return types
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    hit_rate: float
    mrr: float
    faithfulness: float
    answer_relevance: float
```

### 5.2 Static Analysis Configuration

```toml
# pyproject.toml - mypy configuration
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
```

### 5.3 Type Checking Commands

```bash
# Run mypy on entire codebase
mypy src/ tests/

# Run with strict mode
mypy --strict src/
```

---

## 6. DOCUMENTATION STANDARDS

### 6.1 Docstring Template (Google Style)

```python
def function_name(param1: type, param2: type = default) -> ReturnType:
    """One-line summary of what the function does.
    
    More detailed explanation if necessary. Can span multiple lines.
    Explain the algorithm or approach if non-obvious.
    
    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to default.
    
    Returns:
        Description of return value.
    
    Raises:
        ValueError: When param1 is invalid.
        FileNotFoundError: When file doesn't exist.
    
    Note:
        Any additional notes about usage or limitations.
    
    Example:
        >>> result = function_name("value", 42)
        >>> print(result)
        expected_output
    """
```

### 6.2 Module Documentation

```python
"""Semantic chunking for legal documents.

This module implements semantic chunking based on sentence similarity,
specifically designed for Indonesian legal texts (KUHP).

Attributes:
    DEFAULT_THRESHOLD: Default cosine similarity threshold (0.75).
    DEFAULT_EMBEDDING_MODEL: Default embedding model name.
    
Example:
    >>> from src.preprocessing.chunker import semantic_chunking
    >>> chunks = semantic_chunking(text, threshold=0.75)
"""

from pathlib import Path
from typing import Optional

DEFAULT_THRESHOLD: float = 0.75
DEFAULT_EMBEDDING_MODEL: str = "indobenchmark/indobert-base-p1"

__all__ = ["semantic_chunking", "fixed_size_chunking"]
```

### 6.3 README Requirements

```markdown
# Project Name

## Description
One paragraph describing the project.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from src.retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever()
results = retriever.search("What is theft?")
```

## Testing
```bash
pytest tests/ -v
```

## License
MIT
```

---

## 7. ERROR HANDLING & LOGGING

### 7.1 Exception Hierarchy

```python
# src/utils/exceptions.py

class RAGException(Exception):
    """Base exception for RAG system."""
    pass

class CorpusError(RAGException):
    """Raised when corpus loading/processing fails."""
    pass

class RetrievalError(RAGException):
    """Raised when retrieval fails."""
    pass

class GenerationError(RAGException):
    """Raised when LLM generation fails."""
    pass

class EvaluationError(RAGException):
    """Raised when evaluation fails."""
    pass
```

### 7.2 Logging Standards

```python
# src/utils/logger.py
from loguru import logger
import sys

def setup_logger(log_level: str = "INFO") -> None:
    """Configure application logger."""
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=log_level,
    )
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
        level="DEBUG",
    )

# Usage in modules
from src.utils.logger import logger

logger.info(f"Loading corpus from {corpus_path}")
logger.debug(f"BM25 scores: {scores[:5]}")
logger.error(f"Failed to load FAISS index: {e}", exc_info=True)
```

### 7.3 Error Handling Patterns

```python
# GOOD - Specific exception handling
try:
    result = retriever.search(query)
except RetrievalError as e:
    logger.error(f"Retrieval failed: {e}")
    raise SystemExit(1) from e
except CorpusError as e:
    logger.critical(f"Corpus not loaded: {e}")
    raise SystemExit(1) from e

# BAD - Bare except
try:
    result = retriever.search(query)
except:
    print("Error occurred")
```

---

## 8. TESTING STANDARDS

### 8.1 Test Structure

```python
# tests/test_retrieval.py
import pytest
from src.retrieval.bm25_retriever import BM25Retriever

class TestBM25Retriever:
    """Test suite for BM25Retriever."""
    
    @pytest.fixture
    def retriever(self) -> BM25Retriever:
        """Create BM25Retriever instance for testing."""
        return BM25Retriever(corpus_path=Path("tests/fixtures/sample_corpus.json"))
    
    def test_search_returns_results(self, retriever: BM25Retriever) -> None:
        """Test that search returns non-empty results."""
        results = retriever.search("pencurian", top_k=5)
        assert len(results) > 0
    
    def test_search_empty_query_raises(self, retriever: BM25Retriever) -> None:
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.search("")
    
    def test_search_top_k_validation(self, retriever: BM25Retriever) -> None:
        """Test that negative top_k raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            retriever.search("pencurian", top_k=-1)
```

### 8.2 Test Naming Convention

```python
# Pattern: test_{function}_{scenario}_{expected}
def test_retrieve_context_with_valid_query_returns_results():
    ...

def test_retrieve_context_with_empty_query_raises_valueerror():
    ...

def test_calculate_mrr_with_single_relevant_document_returns_correct_score():
    ...
```

### 8.3 Coverage Requirements

| Module | Minimum Coverage |
|--------|------------------|
| `preprocessing/` | 90% |
| `retrieval/` | 90% |
| `generation/` | 85% |
| `evaluation/` | 90% |
| `ui/` | 70% (UI logic) |

```bash
# Run tests with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Coverage configuration in pyproject.toml
[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/__init__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
]
```

---

## 9. GIT WORKFLOW & COMMIT CONVENTION

### 9.1 Branch Naming

| Branch Type | Pattern | Contoh |
|-------------|---------|--------|
| **Main** | `main` | `main` |
| **Develop** | `develop` | `develop` |
| **Feature** | `feature/{description}` | `feature/semantic-chunking` |
| **Bugfix** | `bugfix/{description}` | `bugfix/crag-threshold` |
| **Hotfix** | `hotfix/{description}` | `hotfix/fix-memory-leak` |
| **Release** | `release/{version}` | `release/v1.0.0` |

### 9.2 Commit Convention (Conventional Commits)

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
| Type | Deskripsi |
|------|-----------|
| `feat` | Fitur baru |
| `fix` | Bug fix |
| `docs` | Perubahan dokumentasi |
| `style` | Perubahan formatting (tidak mengubah logika) |
| `refactor` | Refactoring kode |
| `perf` | Peningkatan performa |
| `test` | Menambah/mengubah test |
| `chore` | Maintenance, dependency updates |

**Scopes:**
| Scope | Deskripsi |
|-------|-----------|
| `preprocessing` | Modul preprocessing |
| `retrieval` | Modul retrieval |
| `generation` | Modul generasi LLM |
| `evaluation` | Modul evaluasi |
| `ui` | Modul antarmuka |
| `config` | Konfigurasi |
| `deps` | Dependencies |

**Contoh:**
```bash
feat(retrieval): add Reciprocal Rank Fusion implementation
fix(evaluation): handle empty context in RAGAS metrics
docs(ui): update Streamlit sidebar documentation
refactor(preprocessing): extract cleaning logic to separate module
perf(retrieval): optimize BM25 tokenization with caching
test(retrieval): add unit tests for hybrid retriever
chore(deps): update langchain to 0.1.0
```

### 9.3 Pull Request Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

---

## 10. DEPENDENCY MANAGEMENT

### 10.1 Version Pinning Strategy

| Dependency Type | Strategy | Contoh |
|-----------------|----------|--------|
| **Critical** | Exact pin (`==`) | `streamlit==1.28.0` |
| **Important** | Minimum version (`>=`) | `sentence-transformers>=2.2.2` |
| **Optional** | Range (`>=,<`) | `pandas>=2.0.0,<3.0.0` |

### 10.2 requirements.txt Organization

```txt
# ============================================================================
# CORE FRAMEWORK
# ============================================================================
streamlit==1.28.0
langchain==0.1.0
langchain-community==0.0.20
langchain-huggingface==0.0.3

# ============================================================================
# NLP & EMBEDDINGS
# ============================================================================
sentence-transformers==2.2.2
transformers==4.36.0
torch==2.1.0
spacy==3.7.2

# ============================================================================
# VECTOR DATABASE
# ============================================================================
faiss-cpu==1.7.4
chromadb==0.4.18

# ============================================================================
# RETRIEVAL
# ============================================================================
rank-bm25==0.2.2
numpy==1.24.3

# ============================================================================
# LLM INFERENCE
# ============================================================================
ollama==0.1.6

# ============================================================================
# EVALUATION
# ============================================================================
ragas==0.1.0
datasets==2.14.0
pandas==2.1.4

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
PyMuPDF==1.23.8

# ============================================================================
# UTILITIES
# ============================================================================
python-dotenv==1.0.0
loguru==0.7.2
pydantic==2.5.3
tenacity==8.2.3
```

---

## 11. SECURITY STANDARDS

### 11.1 Secret Management

```python
# BAD - Hardcoded secrets
API_KEY = "sk-1234567890abcdef"
DB_PASSWORD = "password123"

# GOOD - Environment variables
from src.config.settings import settings

api_key = settings.OPENAI_API_KEY
db_password = settings.DATABASE_PASSWORD
```

### 11.2 Input Validation

```python
from pydantic import BaseModel, validator

class QueryRequest(BaseModel):
    """Validated query request."""
    query: str
    top_k: int = 10
    
    @validator("query")
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    @validator("top_k")
    def top_k_must_be_positive(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError("top_k must be between 1 and 100")
        return v
```

### 11.3 Dependency Security

```bash
# Check for vulnerabilities
pip-audit

# Update dependencies safely
pip install --upgrade streamlit
pip freeze > requirements.txt
```

---

## 12. PERFORMANCE GUIDELINES

### 12.1 Caching Strategy

```python
from functools import lru_cache
import streamlit as st

# GOOD - Cache expensive operations
@st.cache_resource
def load_embedding_model() -> HuggingFaceEmbeddings:
    """Load and cache embedding model."""
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

@lru_cache(maxsize=128)
def compute_similarity(query_hash: str, doc_hash: str) -> float:
    """Cache similarity computations."""
    return cosine_similarity(query_emb, doc_emb)

# BAD - No caching for expensive operations
def load_model():
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
```

### 12.2 Memory Management

```python
# GOOD - Explicit cleanup for large objects
def process_large_corpus(path: Path) -> None:
    """Process corpus with explicit memory management."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    try:
        # Process data
        process(data)
    finally:
        del data  # Explicit cleanup
        import gc
        gc.collect()

# BAD - Memory leak risk
def process_large_corpus(path: Path):
    data = json.load(open(path))
    process(data)
    # data never cleaned up
```

### 12.3 Batch Processing

```python
# GOOD - Batch processing with progress
from tqdm import tqdm

def batch_evaluate(queries: list[str], batch_size: int = 10) -> pd.DataFrame:
    """Evaluate queries in batches."""
    results = []
    for i in tqdm(range(0, len(queries), batch_size)):
        batch = queries[i:i + batch_size]
        batch_results = evaluate_batch(batch)
        results.extend(batch_results)
    return pd.DataFrame(results)
```

---

## 13. CODE REVIEW CHECKLIST

### 13.1 Before Submitting PR

- [ ] Code follows PEP 8 (verified with `flake8`)
- [ ] Type hints added for all functions
- [ ] Docstrings added for public functions/classes
- [ ] Unit tests added for new functionality
- [ ] Tests pass locally (`pytest tests/`)
- [ ] No new warnings introduced (`mypy src/`)
- [ ] Dependencies pinned in `requirements.txt`
- [ ] No hardcoded secrets or credentials
- [ ] No `print()` statements (use `logger` instead)
- [ ] No commented-out code

### 13.2 Review Focus Areas

| Area | Checklist |
|------|-----------|
| **Correctness** | Logic is correct, edge cases handled |
| **Performance** | No unnecessary computations, caching used |
| **Security** | Input validated, no injection risks |
| **Maintainability** | Clear naming, single responsibility |
| **Testability** | Functions are pure where possible, dependencies injected |

---

## 14. ENFORCEMENT

### 14.1 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        args: [--line-length=100]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black]
  
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --ignore=E203,W503]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

### 14.2 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install black isort flake8 mypy
      - run: black --check src/
      - run: isort --check-only src/
      - run: flake8 src/
      - run: mypy src/
  
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/ --cov=src
```

---

## 15. REFERENSI

- PEP 8 - Style Guide for Python Code: https://peps.python.org/pep-0008/
- Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
- Black Formatter: https://black.readthedocs.io/
- mypy Documentation: https://mypy.readthedocs.io/
- Conventional Commits: https://www.conventionalcommits.org/

---

**Dokumen ini wajib diikuti oleh semua kontributor proyek.**
**Pelanggaran terhadap style guide ini harus diperbaiki sebelum merge.**
