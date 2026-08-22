# MIGRATION GUIDE
## From Flat Structure to Modular Architecture

---

## DOKUMEN PENGENDALI VERSI

| Versi | Tanggal | Penulis | Deskripsi |
|-------|---------|---------|-----------|
| 1.0 | 2026-08-13 | Rainova Rahaniawan (152023007) | Panduan migrasi dari struktur lama ke struktur baru |

---

## DAFTAR ISI

1. [Overview](#1-overview)
2. [File Mapping](#2-file-mapping)
3. [Step-by-Step Migration](#3-step-by-step-migration)
4. [Breaking Changes](#4-breaking-changes)
5. [Verification](#5-verification)

---

## 1. OVERVIEW

### 1.1 Before (Flat Structure)

```
File Hukum/
├── fase1_cleaning.py
├── fase2_faiss_indexer.py
├── fase3_hybrid_crag.py
├── fase4_mass_evaluation.py
├── fase5_ragas_evaluation.py
├── fase6_streamlit_app.py
├── kuhp_bersih.json
├── golden_dataset_rag_hukum_indonesia_rev3.csv
├── faiss_index_kuhp/
├── chroma_db_kuhp/
├── hasil_akhir_ragas_skripsi.csv
└── hasil_generasi_llama3.csv
```

### 1.2 After (Modular Structure)

```
File Hukum/
├── src/
│   ├── preprocessing/
│   │   ├── cleaner.py
│   │   ├── normalizer.py
│   │   └── chunker.py
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   ├── hybrid_retriever.py
│   │   └── reranker.py
│   ├── generation/
│   │   ├── llm.py
│   │   └── prompts.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── golden_dataset.py
│   │   └── runner.py
│   ├── ui/
│   │   ├── app.py
│   │   ├── components.py
│   │   └── styles.py
│   └── utils/
│       ├── logger.py
│       ├── validators.py
│       ├── helpers.py
│       └── exceptions.py
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_indexing.py
│   ├── run_evaluation.py
│   └── run_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── indexes/
│   └── datasets/
└── tests/
```

---

## 2. FILE MAPPING

### 2.1 Source Code Mapping

| Old File | New Location | Key Changes |
|----------|--------------|-------------|
| `fase1_cleaning.py` | `src/preprocessing/cleaner.py`, `normalizer.py`, `chunker.py` | Split into 3 focused modules |
| `fase2_faiss_indexer.py` | `src/retrieval/dense_retriever.py` | Added class-based interface |
| `fase3_hybrid_crag.py` | `src/retrieval/hybrid_retriever.py` + `reranker.py` | Separated RRF and CRAG logic |
| `fase4_mass_evaluation.py` | `src/evaluation/runner.py` | Modularized with proper error handling |
| `fase5_ragas_evaluation.py` | `src/evaluation/metrics.py` + `runner.py` | Integrated into unified runner |
| `fase6_streamlit_app.py` | `src/ui/app.py` + `components.py` + `styles.py` | Separated UI components |

### 2.2 Data File Mapping

| Old Location | New Location | Action Required |
|--------------|--------------|-----------------|
| `kuhp_bersih.json` | `data/processed/kuhp_bersih.json` | **Move file** |
| `golden_dataset_rag_hukum_indonesia_rev3.csv` | `data/datasets/golden_dataset_rag_hukum_indonesia_rev3.csv` | **Move file** |
| `faiss_index_kuhp/` | `data/indexes/faiss_index_kuhp/` | **Move directory** |
| `chroma_db_kuhp/` | `data/indexes/chroma_db_kuhp/` | **Move directory** |
| `hasil_akhir_ragas_skripsi.csv` | `logs/evaluation_results.csv` | **Move file** |
| `hasil_generasi_llama3.csv` | `logs/generation_results.csv` | **Move file** |

### 2.3 Runner Script Mapping

| Old Command | New Command | Notes |
|-------------|--------------|-------|
| `python fase1_cleaning.py` | `python scripts/run_preprocessing.py` | Includes all 3 steps |
| `python fase2_faiss_indexer.py` | `python scripts/run_indexing.py` | Uses new DenseRetriever |
| `python fase4_mass_evaluation.py` | `python scripts/run_evaluation.py` | Unified evaluation |
| `streamlit run fase6_streamlit_app.py` | `streamlit run src/ui/app.py` or `python scripts/run_app.py` | Both work |

---

## 3. STEP-BY-STEP MIGRATION

### Step 1: Create Directory Structure

```bash
cd "File Hukum"
mkdir -p src/{config,preprocessing,retrieval,generation,evaluation,ui,utils}
mkdir -p tests
mkdir -p data/{raw,processed,indexes,datasets}
mkdir -p scripts
mkdir -p docs
mkdir -p logs
```

### Step 2: Move Data Files

```bash
# Move processed data
mv kuhp_bersih.json data/processed/

# Move datasets
mv golden_dataset_rag_hukum_indonesia_rev3.csv data/datasets/

# Move indexes
mv faiss_index_kuhp data/indexes/
mv chroma_db_kuhp data/indexes/

# Move results
mv hasil_akhir_ragas_skripsi.csv logs/
mv hasil_generasi_llama3.csv logs/
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Optional, for development
```

### Step 4: Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### Step 5: Verify Installation

```bash
# Run tests
pytest tests/ -v

# Verify imports
python -c "from src.retrieval.hybrid_retriever import HybridRetriever; print('OK')"
```

### Step 6: Run Application

```bash
# Option 1: Direct Streamlit
streamlit run src/ui/app.py

# Option 2: Runner script
python scripts/run_app.py
```

---

## 4. BREAKING CHANGES

### 4.1 Import Paths

**Before:**
```python
# Old - flat structure
from fase3_hybrid_crag import reciprocal_rank_fusion
```

**After:**
```python
# New - modular structure
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import ReRanker
```

### 4.2 Configuration

**Before:**
```python
# Hardcoded paths
with open("kuhp_bersih.json") as f:
    data = json.load(f)
vectorstore = FAISS.load_local("faiss_index_kuhp", ...)
```

**After:**
```python
# Centralized configuration
from src.config.settings import settings

with open(settings.processed_corpus_path) as f:
    data = json.load(f)
vectorstore = FAISS.load_local(str(settings.faiss_index_path), ...)
```

### 4.3 Error Handling

**Before:**
```python
# Silent failures
try:
    result = retriever.search(query)
except:
    print("Error")
```

**After:**
```python
# Explicit exceptions
from src.utils.exceptions import RetrievalError

try:
    result = retriever.search(query)
except RetrievalError as e:
    logger.error(f"Retrieval failed: {e}")
    raise
```

### 4.4 Logging

**Before:**
```python
print("[1/4] Loading data...")
print(f"Loaded {len(data)} documents")
```

**After:**
```python
from src.utils.logger import get_logger
logger = get_logger(__name__)

logger.info("Loading data...")
logger.info(f"Loaded {len(data)} documents")
```

---

## 5. VERIFICATION

### 5.1 Verification Checklist

- [ ] All data files moved to `data/` subdirectories
- [ ] All Python files present in `src/` submodules
- [ ] `pip install -r requirements.txt` succeeds
- [ ] `pytest tests/ -v` passes
- [ ] `streamlit run src/ui/app.py` launches successfully
- [ ] `python scripts/run_preprocessing.py` works
- [ ] `python scripts/run_indexing.py` works
- [ ] `python scripts/run_evaluation.py` works

### 5.2 Quick Verification Script

```bash
# Run this script to verify migration
python scripts/verify_migration.py
```

Expected output:
```
✅ Directory structure created
✅ Dependencies installed
✅ Data files migrated
✅ Tests pass
✅ Application launches successfully

Migration complete!
```

---

## 6. ROLLBACK PLAN

If issues arise during migration:

```bash
# 1. Restore old files from git
git checkout fase1_cleaning.py fase2_faiss_indexer.py ...

# 2. Move data files back
mv data/processed/kuhp_bersih.json ./
mv data/datasets/golden_dataset_rag_hukum_indonesia_rev3.csv ./
mv data/indexes/faiss_index_kuhp ./
mv data/indexes/chroma_db_kuhp ./
mv logs/*.csv ./

# 3. Use old commands
python fase6_streamlit_app.py
```

---

**Dokumen ini disusun untuk memudahkan migrasi ke struktur modular yang lebih maintainable.**
