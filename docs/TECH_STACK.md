# TECH STACK DEFINITION
## Sistem Asisten Regulasi RAG untuk Domain Hukum Indonesia
### Hybrid Retrieval dan Semantic Chunking pada KUHP Baru (UU No. 1 Tahun 2023)

---

## DOKUMEN PENGENDALI VERSI

| Versi | Tanggal | Penulis | Deskripsi |
|-------|---------|---------|-----------|
| 1.0 | 2026-08-13 | Rainova Rahaniawan (152023007) | Definisi teknis lengkap untuk arsitektur sistem RAG |

---

## DAFTAR ISI

1. [Visi Arsitektur](#1-visi-arsitektur)
2. [Bahasa Pemrograman](#2-bahasa-pemrograman)
3. [Core Framework & Libraries](#3-core-framework--libraries)
4. [Dependency Management](#4-dependency-management)
5. [Environment & Tooling](#5-environment--tooling)
6. [Hardware Specifications](#6-hardware-specifications)
7. [Deployment Architecture](#7-deployment-architecture)
8. [Version Pinning Strategy](#8-version-pinning-strategy)
9. [Migration Path](#9-migration-path)

---

## 1. VISI ARSITEKTUR

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SISTEM ASISTEN REGULASI RAG                       │
│                     Arsitektur: Monolithic Streamlit                 │
└─────────────────────────────────────────────────────────────────────┘

Layer 1: Presentation Layer
  └── Streamlit Web UI (Single Page Application)
      ├── Chat Interface
      ├── Configuration Panels
      └── Evaluation Dashboard

Layer 2: Application Layer
  └── Python Business Logic
      ├── Preprocessing Pipeline
      ├── Retrieval Pipeline (Hybrid + RRF)
      ├── Generation Pipeline (LLM Inference)
      └── Evaluation Pipeline (RAGAS)

Layer 3: Data Layer
  ├── File-Based Storage (JSON, CSV)
  ├── Vector Database (FAISS / Chroma)
  └── Local LLM Runtime (Ollama)

Layer 4: Infrastructure Layer
  ├── Hardware: Asus Vivobook Pro 14 OLED
  ├── OS: Windows 11 / Linux
  └── Optional: BRIN HPC (untuk skala besar)
```

---

## 2. BAHASA PEMROGRAMAN

| Aspek | Spesifikasi | Justifikasi |
|-------|-------------|-------------|
| **Primary Language** | Python 3.9 - 3.11 | Compatibility dengan library NLP/ML, sesuai proposal |
| **Minimum Version** | Python 3.9 | Support untuk type hints modern dan performance improvements |
| **Recommended Version** | Python 3.10 atau 3.11 | Stability + performance (faster execution) |
| **Type System** | Static typing (mypy) | Maintainability untuk thesis yang kompleks |
| **Code Style** | PEP 8 + Black formatter | Konsistensi kode |

**Verifikasi Instalasi**:
```bash
python --version  # Harus >= 3.9
pip --version
```

---

## 3. CORE FRAMEWORK & LIBRARIES

### 3.1 Backend & Application Framework

| Library | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| **Streamlit** | >=1.28.0 | Web UI framework | Single-file deployment, native Python integration, ideal untuk research prototype |
| **LangChain** | >=0.1.0 | LLM orchestration | Standard untuk RAG pipeline, abstraksi yang bersih |
| **LangChain Community** | >=0.0.20 | Integrasi lokal | BM25, FAISS, Ollama connectors |
| **LangChain HuggingFace** | >=0.0.3 | Embedding models | Interface untuk model embedding |

**Alternatif yang Ditolak**:
- FastAPI/Flask: Overkill untuk single-user research prototype
- Django: Terlalu berat, tidak cocok untuk ML pipeline
- Gradio: Kurang fleksibel untuk UI kompleks

### 3.2 Natural Language Processing

| Library | Version | Purpose | Justifikasi |
|---------|---------|---------|---------------|
| **Sentence-Transformers** | >=2.2.0 | Embedding generation | State-of-the-art untuk multilingual Indonesian |
| **Transformers (HuggingFace)** | >=4.35.0 | Model loading | Interface untuk Indobenchmark/IndoBERT |
| **spaCy** | >=3.6.0 | Sentence splitting | Fast, accurate untuk Bahasa Indonesia |
| **NLTK** | >=3.8.0 | Text preprocessing | Tokenization, stopwords removal |

### 3.3 Vector Database

| Library | Version | Purpose | Justifikasi |
|---------|---------|---------|---------------|
| **FAISS (faiss-cpu)** | >=1.7.4 | Vector similarity search | Industry standard, file-based, no server required |
| **ChromaDB** | >=0.4.0 | Alternative vector DB | Persistence bawaan, API yang lebih simpel |

**Pilihan**: FAISS untuk production, Chroma untuk development cepat.

### 3.4 Retrieval & Ranking

| Library | Version | Purpose | Justifikasi |
|---------|---------|---------|---------------|
| **Rank-BM25** | >=0.2.2 | Sparse retrieval | Pure Python implementation, cepat, akurat |
| **NumPy** | >=1.24.0 | Numerical operations | Cosine similarity, array operations |
| **SciPy** | >=1.10.0 | Scientific computing | Optional: advanced similarity metrics |

### 3.5 LLM & Inference

| Library | Version | Purpose | Justifikasi |
|---------|---------|---------|---------------|
| **Ollama** | >=0.1.0 | Local LLM runtime | Menjalankan Llama-3 8B secara lokal |
| **LangChain Ollama** | >=0.0.3 | LLM interface | Integration dengan LangChain pipeline |
| **Transformers** | >=4.35.0 | Alternative inference | Untuk model HuggingFace langsung |

**Model yang Digunakan**:
- **LLM**: Llama-3 8B Instruct (via Ollama)
- **Quantization**: 4-bit GGUF/AWQ
- **Embedding**: `indobenchmark/indobert-base-p1` (dengan mean pooling)

### 3.6 Evaluation & Metrics

| Library | Version | Purpose | Justifikasi |
|---------|---------|---------|---------------|
| **RAGAS** | >=0.1.0 | RAG evaluation framework | Faithfulness, Answer Relevance metrics |
| **Datasets (HuggingFace)** | >=2.14.0 | Dataset handling | Format untuk RAGAS evaluation |
| **Pandas** | >=2.0.0 | Data manipulation | CSV processing, result aggregation |
| **Scikit-learn** | >=1.3.0 | Evaluation metrics | Optional: precision, recall, F1 |

### 3.7 Document Processing

| Library | Version | Purpose | Justifikasi |
|---------|---------|---------|---------------|
| **PyMuPDF (fitz)** | >=1.23.0 | PDF extraction | Paling akurat untuk PDF UU Indonesia |
| **python-docx** | >=0.8.11 | DOCX processing | Alternatif format dokumen |
| **python-pptx** | >=0.6.23 | PowerPoint processing | Untuk presentasi proposal |
| **Pillow** | >=10.0.0 | Image processing | Diagram, screenshot handling |

### 3.8 Utilities & Configuration

| Library | Version | Purpose | Justifikasi |
|---------|---------|---------|---------------|
| **python-dotenv** | >=1.0.0 | Environment variables | Configuration management |
| **PyYAML** | >=6.0 | YAML parsing | Configuration files |
| **tqdm** | >=4.66.0 | Progress bars | Batch processing visibility |
| **loguru** | >=0.7.0 | Logging | Better logging than standard library |
| **pydantic** | >=2.0.0 | Data validation | Configuration schemas |
| **tenacity** | >=8.2.0 | Retry logic | Robust API calls |

---

## 4. DEPENDENCY MANAGEMENT

### 4.1 File Structure

```
File Hukum/
├── requirements.txt          # Pinned dependencies
├── requirements-dev.txt      # Development tools
├── pyproject.toml           # Modern Python packaging
├── .python-version          # Python version pinning
├── venv/                    # Virtual environment (gitignored)
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── retrieval.py
    ├── generation.py
    ├── evaluation.py
    └── ui.py
```

### 4.2 requirements.txt (Complete)

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
accelerate==0.25.0
spacy==3.7.2
nltk==3.8.1

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
scipy==1.11.4

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
scikit-learn==1.3.2

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
PyMuPDF==1.23.8
python-docx==0.8.11
python-pptx==0.6.23
Pillow==10.1.0

# ============================================================================
# UTILITIES
# ============================================================================
python-dotenv==1.0.0
pyyaml==6.0.1
tqdm==4.66.1
loguru==0.7.2
pydantic==2.5.3
tenacity==8.2.3

# ============================================================================
# TESTING (Development)
# ============================================================================
pytest==7.4.3
pytest-cov==4.1.0
```

### 4.3 requirements-dev.txt (Development Only)

```txt
# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-mock==3.12.0

# Code Quality
black==23.12.0
flake8==6.1.0
mypy==1.7.1
pylint==3.0.3

# Documentation
sphinx==7.2.6
sphinx-rtd-theme==1.3.0

# Jupyter (for experimentation)
jupyter==1.0.0
ipykernel==6.27.1
matplotlib==3.8.2
seaborn==0.13.0
```

### 4.4 pyproject.toml (Modern Alternative)

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rag-asisten-regulasi"
version = "1.0.0"
description = "Sistem Asisten Regulasi RAG untuk KUHP Baru"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.9"
authors = [
    {name = "Rainova Rahaniawan", email = "152023007@itenas.edu.id"}
]
keywords = ["rag", "legal", "indonesian", "kuihp", "llm"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "streamlit>=1.28.0",
    "langchain>=0.1.0",
    "langchain-community>=0.0.20",
    "sentence-transformers>=2.2.2",
    "transformers>=4.36.0",
    "torch>=2.1.0",
    "faiss-cpu>=1.7.4",
    "rank-bm25>=0.2.2",
    "ollama>=0.1.6",
    "ragas>=0.1.0",
    "pandas>=2.1.4",
    "PyMuPDF>=1.23.8",
    "loguru>=0.7.2",
    "pydantic>=2.5.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.3",
    "black>=23.12.0",
    "mypy>=1.7.1",
]

[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
```

---

## 5. ENVIRONMENT & TOOLING

### 5.1 Development Environment

| Tool | Version | Purpose | Justifikasi |
|------|---------|---------|-------------|
| **IDE/Editor** | VS Code 1.80+ | Primary development | Python extension, debugging, Git integration |
| **Git** | 2.40+ | Version control | Thesis progress tracking |
| **Ollama** | 0.1.6+ | LLM runtime | Local inference untuk Llama-3 |
| **Postman/Thunder Client** | Latest | API testing | Jika diperlukan untuk debugging |

### 5.2 Virtual Environment

```bash
# Python 3.9+ required
python -m venv venv

# Activation (Windows)
venv\Scripts\activate

# Activation (Linux/Mac)
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import streamlit; print(streamlit.__version__)"
```

### 5.3 Environment Variables (.env)

```env
# Application
APP_NAME=Asisten Hukum AI (RAG KUHP)
APP_VERSION=1.0.0
ENVIRONMENT=development  # development | production

# Paths
CORPUS_PATH=../data/raw/KUHP BARU UU Nomor 1 Tahun 2023.pdf
PROCESSED_CORPUS_PATH=../data/processed/kuhp_bersih.json
FAISS_INDEX_PATH=../data/indexes/faiss_index_kuhp
CHROMA_DB_PATH=../data/indexes/chroma_db_kuhp
GOLDEN_DATASET_PATH=../data/datasets/golden_dataset_rag_hukum_indonesia_rev3.csv

# Models
EMBEDDING_MODEL=indobenchmark/indobert-base-p1
LLM_MODEL=llama3
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=512

# Retrieval
TOP_K=10
RRF_K=60
CRAG_THRESHOLD=0.020
CHUNKING_THRESHOLD=0.75

# Database (Future)
# DATABASE_URL=postgresql://user:pass@localhost:5432/rag_legal
# REDIS_URL=redis://localhost:6379/0

# Authentication (Future)
# SECRET_KEY=your-secret-key-here
# ALGORITHM=HS256
# ACCESS_TOKEN_EXPIRE_MINUTES=30
```

---

## 6. HARDWARE SPECIFICATIONS

### 6.1 Development Machine (Current)

| Komponen | Spesifikasi | Kebutuhan Minimal | Catatan |
|----------|-------------|-------------------|---------|
| **Laptop** | Asus Vivobook Pro 14 OLED | 16 GB RAM | Sesuai proposal |
| **Processor** | Intel Core i7 / AMD Ryzen 7 | Intel i5 / AMD Ryzen 5 | CPU-bound untuk preprocessing |
| **RAM** | 16 GB DDR4 | 16 GB | Wajib untuk Llama-3 8B 4-bit |
| **Storage** | 512 GB SSD | 256 GB SSD | Korpus + model + vector DB |
| **GPU** | NVIDIA RTX 4050 6GB (Opsional) | CPU only | Accelerates embedding generation |

### 6.2 Inference Requirements

| Model | Quantization | VRAM Required | RAM Fallback |
|-------|--------------|---------------|--------------|
| Llama-3 8B | 4-bit GGUF | ~6 GB | 8 GB system RAM |
| Llama-3 8B | 8-bit AWQ | ~8 GB | 12 GB system RAM |
| IndoBERT Base | Full precision | ~2 GB | 4 GB system RAM |
| BGE-M3 | Full precision | ~2 GB | 4 GB system RAM |

### 6.3 Optional HPC (BRIN)

```bash
# Jika perlu komputasi besar:
# - BRIN HPC cluster
# - NVIDIA A100/H100 GPUs
# - 64+ GB RAM per node
# - Slurm workload manager
```

---

## 7. DEPLOYMENT ARCHITECTURE

### 7.1 Current Architecture (Thesis Scope)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT: SINGLE MACHINE                       │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Browser    │────▶│  Streamlit   │────▶│   Ollama     │
│  (Chrome)    │     │  App (8501)  │     │  Llama-3 8B  │
└──────────────┘     └──────┬───────┘     └──────────────┘
                             │
                    ┌────────┴────────┐
                    │   Local Files    │
                    │  • kuhp_bersih.json │
                    │  • faiss_index/   │
                    │  • golden_dataset │
                    └──────────────────┘

Port: 8501 (Streamlit default)
Access: http://localhost:8501
```

**Startup Command**:
```bash
streamlit run src/ui/app.py --server.port 8501
```

### 7.2 Future Architecture (Post-Thesis)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT: MULTI-USER PRODUCTION                │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Browser    │────▶│   Nginx      │────▶│   FastAPI    │
│  (HTTPS)     │     │  Reverse     │     │   Backend    │
└──────────────┘     │   Proxy      │     └──────┬───────┘
                     └──────────────┘            │
                                                  ▼
                                         ┌──────────────┐
                                         │  PostgreSQL   │
                                         │  + pgvector   │
                                         └──────────────┘
                                                  │
                                                  ▼
                                         ┌──────────────┐
                                         │  Ollama API  │
                                         │  (Llama-3)   │
                                         └──────────────┘
```

---

## 8. VERSION PINNING STRATEGY

### 8.1 Critical Dependencies (Pinned)

```txt
# These versions are tested and verified
streamlit==1.28.0
langchain==0.1.0
faiss-cpu==1.7.4
ollama==0.1.6
ragas==0.1.0
```

### 8.2 Flexible Dependencies (Minimum Version)

```txt
# These can be updated with minor versions
sentence-transformers>=2.2.2
transformers>=4.36.0
pandas>=2.1.4
```

### 8.3 Version Verification Script

```python
# verify_environment.py
import sys
import importlib

REQUIRED_VERSIONS = {
    'streamlit': '1.28.0',
    'langchain': '0.1.0',
    'faiss': '1.7.4',
    'ollama': '0.1.6',
    'ragas': '0.1.0',
    'sentence_transformers': '2.2.2',
}

def verify_environment():
    print("="*60)
    print("ENVIRONMENT VERIFICATION")
    print("="*60)
    
    # Python version
    print(f"\nPython: {sys.version}")
    if sys.version_info < (3, 9):
        print("❌ ERROR: Python 3.9+ required")
        return False
    
    # Library versions
    all_ok = True
    for lib, min_version in REQUIRED_VERSIONS.items():
        try:
            module = importlib.import_module(lib)
            version = getattr(module, '__version__', 'unknown')
            status = "✅" if version >= min_version else "❌"
            print(f"{status} {lib}: {version} (>= {min_version})")
            if version < min_version:
                all_ok = False
        except ImportError:
            print(f"❌ {lib}: NOT INSTALLED")
            all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ All dependencies satisfied")
    else:
        print("❌ Some dependencies missing or outdated")
    print("="*60)
    
    return all_ok

if __name__ == "__main__":
    verify_environment()
```

---

## 9. MIGRATION PATH

### 9.1 Phase 1: Current State (Thesis Implementation)

**Stack**: Python 3.9+ + Streamlit + FAISS + Ollama
**Storage**: File-based (JSON, CSV, FAISS)
**Auth**: None / Lightweight session
**Deployment**: Local single-user

### 9.2 Phase 2: Enhanced Prototype (Post-Thesis)

**Stack**: Same + SQLite / JSON-based RBAC
**Storage**: SQLite for users/logs + File-based for corpus
**Auth**: Session-based with password hashing
**Deployment**: Local network (department use)

### 9.3 Phase 3: Production Ready (Future)

**Stack**: Same + FastAPI + PostgreSQL + Redis
**Storage**: PostgreSQL with pgvector + Redis cache
**Auth**: JWT-based with refresh tokens
**Deployment**: Docker Compose / Cloud

---

## 10. DEVELOPMENT WORKFLOW

### 10.1 Daily Development

```bash
# 1. Activate environment
venv\Scripts\activate

# 2. Start Ollama (separate terminal)
ollama serve

# 3. Pull required models (one-time)
ollama pull llama3
ollama pull llama3:8b-instruct-q4_0

# 4. Run preprocessing (if needed)
python scripts/legacy/fase1_cleaning.py

# 5. Run indexing (if needed)
python scripts/legacy/fase2_faiss_indexer.py

# 6. Start Streamlit app
streamlit run src/ui/app.py

# 7. Run evaluation (separate terminal)
python scripts/legacy/fase4_mass_evaluation.py
python scripts/legacy/fase5_ragas_evaluation.py
```

### 10.2 Testing Workflow

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Linting
flake8 src/
black --check src/
mypy src/
```

### 10.3 Git Workflow

```bash
# Branch naming
main                    # Stable, thesis-ready code
develop                 # Integration branch
feature/semantic-chunking
feature/hybrid-retrieval
bugfix/crag-threshold

# Commit convention
feat: add semantic chunking with threshold 0.75
fix: adjust CRAG threshold for better precision
docs: update specification with RBAC details
test: add unit tests for BM25 retrieval
```

---

## 11. SECURITY CONSIDERATIONS

| Aspek | Implementasi | Justifikasi |
|-------|---------------|-------------|
| **Secret Management** | `.env` file (gitignored) | Tidak ada hardcoded credentials |
| **Password Storage** | scrypt/argon2 hashing | Modern, secure hashing algorithms |
| **Input Sanitization** | Regex validation | Prevent injection attacks |
| **LLM Prompt Injection** | Strict system prompts | Guardrails against jailbreaking |
| **File Upload** | Extension whitelist | Prevent malicious uploads |

---

## 12. MONITORING & LOGGING

### 12.1 Application Logging

```python
from loguru import logger

# Configuration
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="30 days",
    encoding="utf-8",
    level="INFO"
)

# Usage
logger.info(f"User {username} executed query: {query[:50]}...")
logger.error(f"Retrieval failed: {str(e)}")
```

### 12.2 Performance Monitoring

```python
import time
from functools import wraps

def monitor_latency(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency = (time.time() - start) * 1000
        logger.info(f"{func.__name__} took {latency:.2f}ms")
        return result
    return wrapper
```

---

## 13. APPENDIX: QUICK REFERENCE

### 13.1 Installation Script (setup.sh / setup.bat)

```bash
#!/bin/bash
# setup.sh - Automated environment setup

echo "Setting up RAG Asisten Regulasi..."

# Check Python version
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Create directories
mkdir -p logs faiss_index_kuhp chroma_db_kuhp

# Verify installation
python verify_environment.py

echo "Setup complete! Run: streamlit run src/ui/app.py"
```

### 13.2 Tech Stack Summary Table

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Language** | Python | 3.9+ | Core development |
| **UI Framework** | Streamlit | 1.28.0 | Web interface |
| **LLM Framework** | LangChain | 0.1.0 | RAG orchestration |
| **LLM Runtime** | Ollama | 0.1.6 | Local inference |
| **LLM Model** | Llama-3 | 8B-Instruct | Answer generation |
| **Embedding** | Sentence-Transformers | 2.2.2 | Semantic encoding |
| **Vector DB** | FAISS | 1.7.4 | Similarity search |
| **Sparse Retrieval** | Rank-BM25 | 0.2.2 | Keyword matching |
| **Evaluation** | RAGAS | 0.1.0 | Faithfulness metrics |
| **PDF Processing** | PyMuPDF | 1.23.8 | Text extraction |
| **Data Processing** | Pandas | 2.1.4 | Dataset handling |
| **Logging** | Loguru | 0.7.2 | Application logs |
| **Validation** | Pydantic | 2.5.3 | Config schemas |

---

## 14. REFERENSI

- Proposal Tugas Akhir: `152023007_File_Proposal_Rainova_Rahaniawan_rev2.docx`
- Implementasi Existing: `scripts/legacy/fase1_cleaning.py` - `scripts/legacy/fase6_streamlit_app.py`
- Spesifikasi Sistem: `SPECIFICATION.md`
- Assessment Arsitektur: `ASSESSMENT.md`
- Schema Database: `database.sql`

---

**Dokumen ini ditetapkan sebagai referensi teknis resmi untuk pengembangan**
**Sistem Asisten Regulasi RAG - Rainova Rahaniawan (152023007) - ITENAS 2026**
