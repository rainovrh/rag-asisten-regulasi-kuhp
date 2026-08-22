# RAG Asisten Regulasi - KUHP Baru

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-green?logo=langchain)
![FAISS](https://img.shields.io/badge/FAISS-1.7.4%2B-orange?logo=meta)
![Ollama](https://img.shields.io/badge/Ollama-0.1.6%2B-black?logo=ollama)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Sistem Asisten Regulasi berbasis Retrieval-Augmented Generation (RAG)** untuk dokumen hukum Indonesia, khususnya KUHP Baru (UU No. 1 Tahun 2023). Proyek ini mengimplementasikan **Semantic Chunking**, **Hybrid Retrieval** (BM25 + Dense + RRF), dan **CRAG Filtering** untuk mitigasi halusinasi, dengan evaluasi saintifik menggunakan **RAGAS** (Faithfulness & Answer Relevance).

> **Catatan**: Ini adalah proyek portofolio akademis. Hasil analisis AI tidak menggantikan konsultasi dengan ahli hukum.

---

## Fitur Utama

- **Preprocessing Cerdas**: Pembersihan PDF, OCR artifact cleaning, normalisasi teks, dan semantic chunking
- **Hybrid Retrieval**: Kombinasi BM25 (sparse) dan Dense Retrieval dengan Reciprocal Rank Fusion (RRF)
- **CRAG Filtering**: Contextual Retrieval Augmented Generation untuk mitigasi halusinasi
- **Evaluasi Saintifik**: Metrik Hit Rate, MRR, dan RAGAS (Faithfulness, Answer Relevance)
- **Antarmuka Web**: Streamlit chat interface dengan tema akademis Deep Navy + Gold + Ivory
- **Golden Dataset**: 50 skenario kasus hukum (factoid & open-ended) dengan ground truth terverifikasi

---

## Tech Stack

| Layer | Teknologi | Versi |
|-------|-----------|-------|
| **Bahasa Pemrograman** | Python | 3.9 - 3.11 |
| **UI Framework** | Streamlit | >= 1.28.0 |
| **LLM Framework** | LangChain + LangChain Community | >= 0.1.0 |
| **LLM Runtime** | Ollama (Llama-3 8B) | >= 0.1.6 |
| **Embedding** | Sentence-Transformers (MiniLM-L12-v2) | >= 2.2.2 |
| **Vector Database** | FAISS (faiss-cpu) | >= 1.7.4 |
| **Sparse Retrieval** | Rank-BM25 | >= 0.2.2 |
| **Evaluasi** | RAGAS | >= 0.1.0 |
| **PDF Processing** | PyMuPDF | >= 1.23.8 |
| **Data Processing** | Pandas | >= 2.1.4 |
| **Logging** | Loguru | >= 0.7.2 |
| **Validation** | Pydantic | >= 2.5.0 |

---

## Struktur Proyek

```
File Hukum/
├── src/
│   ├── config/             # Konfigurasi (settings, paths, constants)
│   ├── preprocessing/      # PDF cleaning, OCR cleaning, normalization, chunking
│   ├── retrieval/          # BM25, Dense, Hybrid retrieval, RRF, CRAG
│   ├── generation/         # LLM inference & prompt templates
│   ├── evaluation/         # Golden dataset, metrics, runner
│   ├── ui/                 # Streamlit app, components, styles
│   └── utils/              # Logger, validators, helpers, exceptions
├── scripts/
│   ├── run_app.py          # Entry point Streamlit
│   ├── run_preprocessing.py # Build corpus dari PDF
│   ├── run_indexing.py     # Build FAISS index
│   ├── run_evaluation.py   # Jalankan evaluasi batch
│   └── legacy/             # Script legacy (fase1-fase6)
├── data/
│   ├── raw/                # PDF asli KUHP
│   ├── processed/          # kuhp_bersih.json, pasal_to_chunks.json
│   ├── indexes/            # FAISS index
│   └── datasets/           # Golden dataset CSV + XLSX
├── tests/                  # Unit tests
├── docs/                   # Dokumentasi teknis
│   ├── proposal/
│   ├── presentations/
│   └── diagrams/
├── logs/                   # Application logs
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

---

## Instalasi

### Prerequisites

- Python 3.9 atau lebih tinggi
- Ollama (untuk inferensi Llama-3 lokal)
- Git (opsional, untuk cloning)

### Langkah-langkah

```bash
# 1. Clone repository
git clone https://github.com/rainovrh/rag-asisten-regulasi-kuhp.git
cd rag-asisten-regulasi-kuhp

# 2. Buat virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Konfigurasi environment
cp .env.example .env
# Edit .env sesuai kebutuhan

# 5. Setup Ollama
ollama serve
ollama pull llama3
ollama pull llama3:8b-instruct-q4_0

# 6. Jalankan preprocessing (jika belum)
python scripts/run_preprocessing.py
python scripts/run_indexing.py

# 7. Jalankan aplikasi
streamlit run src/ui/app.py
```

---

## Penggunaan

### Mode Aplikasi (Streamlit)

```bash
streamlit run src/ui/app.py
```

Atau menggunakan runner script:

```bash
python scripts/run_app.py
```

### Evaluasi

```bash
# Batch evaluation dengan Golden Dataset
python scripts/run_evaluation.py

# Hasil disimpan di:
# - logs/evaluation_results.csv
# - logs/evaluation_results.json
```

---

## Dokumentasi

- `docs/SPECIFICATION.md` - Spesifikasi sistem lengkap
- `docs/TECH_STACK.md` - Definisi tech stack
- `docs/STYLE_GUIDE.md` - Coding standards
- `docs/ASSESSMENT.md` - Arsitektur assessment
- `docs/database.sql` - Database schema reference (skala masa depan)
- `docs/MIGRATION_GUIDE.md` - Panduan migrasi struktur

---

## Metodologi

### 1. Preprocessing Pipeline

```
PDF Input → Document Cleaning → OCR Artifact Cleaning → Normalization → Semantic Chunking → JSON Corpus
```

### 2. Retrieval Pipeline

```
User Query → BM25 (sparse) + Dense Retrieval → RRF Fusion → CRAG Filtering → Top-K Context
```

### 3. Generation Pipeline

```
Query + Context → Prompt Template → Llama-3 8B (Ollama) → Answer with Citation
```

### 4. Evaluation Pipeline

```
Golden Dataset (50 scenarios) → Batch Retrieval → LLM Generation → Hit Rate, MRR, RAGAS Metrics → CSV/JSON Export
```

---

## Kontributor

- **Rainova Rahaniawan** (152023007) - [@rainovrh](https://github.com/rainovrh) - ITENAS 2026
- **Kilo Code** - AI Assistant

## License

[MIT](LICENSE) © 2026 Rainova Rahaniawan
