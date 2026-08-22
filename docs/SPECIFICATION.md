# SPECIFICATION DOCUMENT
## Sistem Asisten Regulasi RAG untuk Domain Hukum Indonesia
### Hybrid Retrieval dan Semantic Chunking pada KUHP Baru (UU No. 1 Tahun 2023)

---

## DOKUMEN PENGENDALI VERSI

| Versi | Tanggal | Penulis | Deskripsi |
|-------|---------|---------|-----------|
| 1.0 | 2026-08-13 | Rainova Rahaniawan (152023007) | Spesifikasi awal sistem RAG Asisten Regulasi |

---

## DAFTAR ISI

1. [Application Workflow](#1-application-workflow)
2. [Feature Specifications](#2-feature-specifications)
3. [User Access Control (RBAC)](#3-user-access-control-rbac)
4. [Data Integration Logic](#4-data-integration-logic)

---

## 1. APPLICATION WORKFLOW

### 1.1 Overview Arsitektur Sistem

Sistem ini mengimplementasikan arsitektur **Retrieval-Augmented Generation (RAG)** dengan dua komponen utama inovasi:
- **Semantic Chunking** untuk fase pra-pemrosesan dokumen
- **Hybrid Retrieval** (BM25 + Dense Retrieval) untuk fase pencarian

### 1.2 Alur Kerja Pipeline Indeksasi (Offline Processing)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE INDEKSASI DATA                       │
└─────────────────────────────────────────────────────────────────┘

Input: Korpus Regulasi (KUHP Baru UU No. 1 Tahun 2023 - Format PDF/TXT)
  │
  ▼
[1] Document Cleaning
  │   • Penghapusan header/footer dan nomor halaman
  │   • Penggabungan kalimat terputus akibat line break PDF
  │   • Normalisasi whitespace
  │
  ▼
[2] Text Normalization & Case Folding
  │   • Lowercasing seluruh teks
  │   • Ekspansi singkatan hukum (UU → undang-undang, KUHP → kitab undang-undang hukum pidana)
  │   • Preservasi angka dan tanda baca penting
  │
  ▼
[3] Semantic Chunking
  │   • Sentence splitting dengan NLP library untuk Bahasa Indonesia
  │   • Pembentukan embedding sementara per kalimat
  │   • Perhitungan Cosine Similarity antar kalimat berurutan
  │   • Threshold determination (nilai ambang: 0.75)
  │   • Penggabungan kalimat dengan kemiripan ≥ threshold menjadi satu chunk
  │   • Pemisahan chunk baru jika kemiripan < threshold
  │
  ▼
[4] Vector Embedding
  │   • Konversi chunk menjadi representasi vektor berdimensi tinggi
  │   • Model embedding: indobenchmark/indobert-base-p1
  │
  ▼
[5] Vector Database Indexing
  │   • Penyimpanan vektor + metadata (nomor pasal, ayat, sumber)
  │   • Database: FAISS atau Chroma
  │   • Pembangunan indeks untuk optimasi pencarian
  │
  ▼
Output: Knowledge Base (Vector Database) siap untuk inferensi
```

### 1.3 Alur Kerja Pipeline Inferensi (Online Processing)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE INFERENSI SISTEM                     │
└─────────────────────────────────────────────────────────────────┘

Input: User Query (Pertanyaan Hukum dalam Bahasa Indonesia)
  │
  ▼
[1] Query Processing
  │   • Case folding sesuai preprocessing korpus
  │   • Normalisasi singkatan
  │
  ▼
[2] Hybrid Retrieval (Parallel Execution)
  │
  │   ┌─────────────────────────┐    ┌─────────────────────────┐
  │   │   BM25 (Sparse)         │    │   Dense Retrieval       │
  │   │   • Keyword matching    │    │   • Cosine similarity   │
  │   │   • TF-IDF weighting    │    │   • Semantic search     │
  │   │   • Top-K = 10          │    │   • Top-K = 10          │
  │   └───────────┬─────────────┘    └───────────┬─────────────┘
  │               │                             │
  │               ▼                             ▼
  │         [BM25 Results]              [Dense Results]
  │               │                             │
  │               └───────────┬─────────────────┘
  │                           ▼
  │               [3] Reciprocal Rank Fusion (RRF)
  │                   • Konstanta k = 60
  │                   • Formula: RRF(d) = Σ 1/(k + rank)
  │                   • Penggabungan skor dari kedua metode
  │                           │
  │                           ▼
  │               [4] Re-ranked Results (Top-K Final)
  │
  ▼
[5] Context Validation (Decision Node)
  │   • Evaluasi skor RRF terhadap ambang batas relevansi
  │   • Jika tidak ada dokumen dengan skor memadai:
  │     → Return: "Informasi tidak ditemukan dalam konteks regulasi"
  │
  ▼
[6] Augmented Prompt Construction
  │   • Gabungkan Top-K chunks dengan user query
  │   • System prompt: "Anda adalah asisten hukum AI..."
  │   • Instruksi ketat: "Jika jawaban tidak ada di konteks, katakan 'Saya tidak tahu'"
  │
  ▼
[7] LLM Inference (Llama-3 8B, 4-bit Quantization)
  │   • Zero-shot prompting
  │   • Tanpa fine-tuning
  │   • Inference lokal (Asus Vivobook Pro 14 OLED)
  │
  ▼
[8] Output Generation
      • Jawaban hukum berbasis konteks
      • Sitasi pasal asli (retrieved chunks)
      • Metadata sumber regulasi
```

### 1.4 Alur Kerja Evaluasi Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE EVALUASI                             │
└─────────────────────────────────────────────────────────────────┘

Input: Golden Dataset (50+ skenario kasus hukum)
  │
  ▼
[1] Batch Inference
  │   • Proses setiap kueri melalui pipeline RAG
  │   • Simpan hasil prediksi untuk setiap query
  │
  ▼
[2] Retrieval Quality Evaluation
  │   • Hit Rate: % kueri dengan minimal 1 dokumen relevan di Top-K
  │   • MRR: rata-rata inversi peringkat dokumen relevan pertama
  │
  ▼
[3] Answer Reliability Evaluation (RAGAS Framework)
  │   • Faithfulness: proporsi klaim jawaban yang valid terhadap konteks
  │   • Answer Relevance: kesamaan semantik antara pertanyaan asli dan buatan
  │
  ▼
[4] Comparative Analysis
  │   • Perbandingan baseline (Fixed-size + Sparse) vs usulan (Semantic + Hybrid)
  │   • Statistik deskriptif dan inferensial
  │
  ▼
Output: Laporan evaluasi kuantitatif (CSV/Excel)
```

---

## 2. FEATURE SPECIFICATIONS

### 2.1 Modul Manajemen Basis Pengetahuan

| ID Fitur | Nama Fitur | Deskripsi | Prioritas |
|----------|-----------|-----------|-----------|
| F-01 | Upload Korpus Regulasi | Upload dokumen regulasi berformat PDF/Txt untuk diindeks | High |
| F-02 | Document Cleaning | Pembersihan noise struktural (header, footer, nomor halaman) | High |
| F-03 | Text Normalization | Normalisasi teks dan case folding dengan ekspansi singkatan hukum | High |
| F-04 | Semantic Chunking | Pemotongan dokumen berdasarkan kesatuan makna semantik | High |
| F-05 | Fixed-size Chunking | Pemotongan dokumen berdasarkan jumlah token tetap (baseline) | Medium |
| F-06 | Vector Embedding | Konversi chunk menjadi representasi vektor numerik | High |
| F-07 | Vector Database Management | Inisialisasi, penyimpanan, dan indeksasi basis data vektor | High |
| F-08 | Metadata Extraction | Ekstraksi dan penyimpanan metadata pasal/ayat/sumber | Medium |

### 2.2 Modul Pencarian dan Retrieval

| ID Fitur | Nama Fitur | Deskripsi | Prioritas |
|----------|-----------|-----------|-----------|
| F-09 | Sparse Retrieval (BM25) | Pencarian leksikal berbasis keyword matching | High |
| F-10 | Dense Retrieval | Pencarian semantik berbasis cosine similarity vektor | High |
| F-11 | Hybrid Retrieval | Kombinasi paralel sparse dan dense retrieval | High |
| F-12 | Reciprocal Rank Fusion (RRF) | Re-ranking hasil gabungan dengan algoritma RRF (k=60) | High |
| F-13 | Top-K Selection | Seleksi dokumen teratas untuk konteks LLM | High |
| F-14 | Query Expansion | Ekspansi kueri dengan sinonim hukum (opsional) | Low |

### 2.3 Modul Pembangkitan Jawaban

| ID Fitur | Nama Fitur | Deskripsi | Prioritas |
|----------|-----------|-----------|-----------|
| F-15 | Augmented Prompt Construction | Konstruksi prompt dengan konteks regulasi dan instruksi ketat | High |
| F-16 | LLM Inference | Inferensi menggunakan Llama-3 8B (4-bit quantization) | High |
| F-17 | Hallucination Guardrail | Instruksi "Saya tidak tahu" untuk jawaban di luar konteks | High |
| F-18 | Citation Generation | Penyertaan sitasi pasal asli dalam output jawaban | Medium |
| F-19 | Context Window Management | Manajemen batas konteks untuk chunk yang panjang | Medium |

### 2.4 Modul Evaluasi dan Monitoring

| ID Fitur | Nama Fitur | Deskripsi | Prioritas |
|----------|-----------|-----------|-----------|
| F-20 | Hit Rate Calculation | Perhitungan proporsi keberhasilan retrieval dokumen relevan | High |
| F-21 | MRR Calculation | Perhitungan Mean Reciprocal Rank untuk kualitas peringkat | High |
| F-22 | RAGAS Faithfulness | Evaluasi kesetiaan jawaban terhadap konteks dokumen | High |
| F-23 | RAGAS Answer Relevance | Evaluasi relevansi jawaban terhadap pertanyaan pengguna | High |
| F-24 | Golden Dataset Management | Upload dan manajemen dataset evaluasi (CSV/Excel) | High |
| F-25 | Batch Evaluation | Proses evaluasi massal terhadap 50+ skenario kasus | High |
| F-26 | Report Generation | Export laporan evaluasi dalam format CSV/Excel | Medium |
| F-27 | Visualization Dashboard | Grafik perbandingan performa baseline vs usulan | Low |

### 2.5 Modul Antarmuka Pengguna

| ID Fitur | Nama Fitur | Deskripsi | Prioritas |
|----------|-----------|-----------|-----------|
| F-28 | Chat Interface | Antarmuka obrolan untuk input pertanyaan hukum | High |
| F-29 | Query Input | Kolom input teks untuk pertanyaan pengguna | High |
| F-30 | Response Display | Tampilan jawaban dengan format terstruktur dan highlighting | High |
| F-31 | Source Citation Display | Tampilkan potongan pasal asli sebagai landasan jawaban | High |
| F-32 | Processing Status | Indikator status pemrosesan (retrieving, generating, dll) | Medium |
| F-33 | Error Handling | Notifikasi jika informasi tidak ditemukan atau sistem error | Medium |
| F-34 | Sidebar Navigation | Panel informasi: panduan, daftar dokumen, disclaimer | Low |
| F-35 | Disclaimer Display | Peringatan batasan AI dan kebutuhan verifikasi praktisi hukum | Medium |

### 2.6 Modul Konfigurasi Sistem

| ID Fitur | Nama Fitur | Deskripsi | Prioritas |
|----------|-----------|-----------|-----------|
| F-36 | Chunking Parameter Config | Konfigurasi threshold cosine similarity (default: 0.75) | Medium |
| F-37 | Retrieval Parameter Config | Konfigurasi Top-K (default: 10) dan konstanta RRF (default: 60) | Medium |
| F-38 | Model Selection | Pilihan embedding model dan LLM (Llama-3 8B) | Medium |
| F-39 | Quantization Settings | Konfigurasi kuantisasi 4-bit (GGUF/AWQ) | Low |
| F-40 | Evaluation Configuration | Upload Golden Dataset dan pengaturan metrik evaluasi | High |

---

## 3. USER ACCESS CONTROL (RBAC)

### 3.1 Role Definitions

 Sistem ini mengadopsi model **Role-Based Access Control (RBAC)** dengan tiga peran utama:

| Role ID | Nama Role | Deskripsi |
|---------|-----------|-----------|
| R-01 | **Researcher (Peneliti)** | Memiliki akses penuh untuk konfigurasi, eksperimen, dan evaluasi sistem |
| R-02 | **Legal Practitioner (Praktisi Hukum)** | Pengguna akhir yang menggunakan sistem untuk analisis regulasi |
| R-03 | **System Administrator** | Mengelola infrastruktur, deployment, dan maintenance sistem |

### 3.2 Permission Matrix

#### 3.2.1 Researcher (Peneliti)

| Modul | Fitur | Permission |
|-------|-------|------------|
| Knowledge Base Management | F-01 Upload Korpus | CREATE |
| Knowledge Base Management | F-02 Document Cleaning | EXECUTE |
| Knowledge Base Management | F-03 Text Normalization | EXECUTE |
| Knowledge Base Management | F-04 Semantic Chunking | EXECUTE |
| Knowledge Base Management | F-05 Fixed-size Chunking | EXECUTE |
| Knowledge Base Management | F-06 Vector Embedding | EXECUTE |
| Knowledge Base Management | F-07 Vector Database Management | EXECUTE |
| Knowledge Base Management | F-08 Metadata Extraction | EXECUTE |
| Retrieval & Search | F-09 BM25 | EXECUTE |
| Retrieval & Search | F-10 Dense Retrieval | EXECUTE |
| Retrieval & Search | F-11 Hybrid Retrieval | EXECUTE |
| Retrieval & Search | F-12 RRF | EXECUTE |
| Retrieval & Search | F-13 Top-K Selection | EXECUTE |
| Retrieval & Search | F-14 Query Expansion | CONFIGURE |
| Answer Generation | F-15 Augmented Prompt | CONFIGURE |
| Answer Generation | F-16 LLM Inference | EXECUTE |
| Answer Generation | F-17 Hallucination Guardrail | CONFIGURE |
| Answer Generation | F-18 Citation Generation | EXECUTE |
| Answer Generation | F-19 Context Window Management | CONFIGURE |
| Evaluation & Monitoring | F-20 Hit Rate | EXECUTE |
| Evaluation & Monitoring | F-21 MRR | EXECUTE |
| Evaluation & Monitoring | F-22 RAGAS Faithfulness | EXECUTE |
| Evaluation & Monitoring | F-23 RAGAS Answer Relevance | EXECUTE |
| Evaluation & Monitoring | F-24 Golden Dataset | CREATE, READ, UPDATE, DELETE |
| Evaluation & Monitoring | F-25 Batch Evaluation | EXECUTE |
| Evaluation & Monitoring | F-26 Report Generation | EXPORT |
| Evaluation & Monitoring | F-27 Visualization | VIEW |
| UI | F-28 Chat Interface | USE |
| UI | F-29 Query Input | USE |
| UI | F-30 Response Display | VIEW |
| UI | F-31 Source Citation | VIEW |
| UI | F-32 Processing Status | VIEW |
| UI | F-33 Error Handling | VIEW |
| UI | F-34 Sidebar | VIEW |
| UI | F-35 Disclaimer | VIEW |
| Configuration | F-36 Chunking Config | CONFIGURE |
| Configuration | F-37 Retrieval Config | CONFIGURE |
| Configuration | F-38 Model Selection | CONFIGURE |
| Configuration | F-39 Quantization | CONFIGURE |
| Configuration | F-40 Evaluation Config | CONFIGURE |

#### 3.2.2 Legal Practitioner (Praktisi Hukum)

| Modul | Fitur | Permission |
|-------|-------|------------|
| Retrieval & Search | F-09 BM25 | USE (read-only) |
| Retrieval & Search | F-10 Dense Retrieval | USE (read-only) |
| Retrieval & Search | F-11 Hybrid Retrieval | USE (read-only) |
| Retrieval & Search | F-12 RRF | USE (read-only) |
| Retrieval & Search | F-13 Top-K Selection | VIEW |
| UI | F-28 Chat Interface | USE |
| UI | F-29 Query Input | USE |
| UI | F-30 Response Display | VIEW |
| UI | F-31 Source Citation | VIEW |
| UI | F-32 Processing Status | VIEW |
| UI | F-33 Error Handling | VIEW |
| UI | F-34 Sidebar | VIEW |
| UI | F-35 Disclaimer | VIEW |
| Configuration | Semua fitur konfigurasi | NONE |

#### 3.2.3 System Administrator

| Modul | Fitur | Permission |
|-------|-------|------------|
| Knowledge Base Management | Semua fitur | MANAGE |
| Retrieval & Search | Semua fitur | MANAGE |
| Answer Generation | Semua fitur | MANAGE |
| Evaluation & Monitoring | Semua fitur | MANAGE |
| UI | Semua fitur | MANAGE |
| Configuration | Semua fitur | MANAGE |
| System | Deployment, Monitoring, Backup | FULL ACCESS |

### 3.3 Access Control Implementation

#### 3.3.1 Authentication
- **Method:** Session-based authentication dengan credential validation
- **Session Management:** Timeout otomatis setelah 30 menit inaktivitas
- **Password Policy:** Minimal 8 karakter, kombinasi huruf besar, kecil, angka, dan simbol

#### 3.3.2 Authorization
- **Implementation:** Decorator-based access control pada setiap endpoint
- **Default Role:** Guest (hanya dapat melihat disclaimer dan dokumentasi)
- **Role Assignment:** Manual oleh System Administrator

#### 3.3.3 Audit Logging
- **Logged Events:** Login/logout, konfigurasi perubahan, evaluasi yang dijalankan
- **Log Storage:** File-based logging dengan timestamp dan user ID
- **Retention:** 90 hari

---

## 4. DATA INTEGRATION LOGIC

### 4.1 Golden Dataset Structure

Berdasarkan analisis file `golden_dataset_rag_hukum_indonesia_rev3.csv`, struktur dataset evaluasi adalah sebagai berikut:

```
Schema:
┌─────────────┬──────────────────────────────────────────────────────────────┐
│ Kolom       │ Deskripsi                                                    │
├─────────────┼──────────────────────────────────────────────────────────────┤
│ ID          │ Nomor urut skenario (1-50)                                  │
│ Query       │ Pertanyaan hukum dalam bahasa Indonesia                     │
│ Konteks     │ Referensi pasal ideal (e.g., "Pasal 1 ayat (1)")           │
│ Ground Truth│ Jawaban ideal berdasarkan KUHP Baru                         │
└─────────────┴──────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow Integration

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GOLDEN DATASET INTEGRATION                       │
└─────────────────────────────────────────────────────────────────────┘

Phase 1: Dataset Acquisition
  │
  ├── Source: KUHP Baru UU No. 1 Tahun 2023 (Portal JDIH)
  │   └── Format: PDF → Ekstraksi teks → Cleaning → JSON (kuhp_bersih.json)
  │
  ├── Expert Verification
  │   └── Validasi oleh ahli hukum terhadap:
  │       • Ketepatan referensi pasal
  │       • Kesesuaian ground truth dengan teks regulasi
  │       • Cakupan skenario (factoid dan open-ended)
  │
  └── Dataset Compilation
      └── 50+ skenario kasus dengan variasi:
          • Tipe factoid (jawaban spesifik)
          • Tipe open-ended (penjelasan konseptual)

Phase 2: Dataset Preprocessing
  │
  ├── CSV Parsing
  │   └── Load golden_dataset_rag_hukum_indonesia_rev3.csv
  │
  ├── Query Normalization
  │   └── Terapkan document cleaning dan case folding yang sama
  │       dengan korpus regulasi untuk konsistensi
  │
  └── Ground Truth Validation
      └── Cross-check dengan kuhp_bersih.json untuk memastikan
          kesesuaian referensi pasal

Phase 3: Evaluation Pipeline Integration
  │
  ├── Batch Processing
  │   └── Iterasi melalui setiap baris Golden Dataset:
  │       1. Ambil Query
  │       2. Jalankan Hybrid Retrieval
  │       3. Hitung Hit Rate dan MRR
  │       4. Generate jawaban via LLM
  │       5. Evaluasi Faithfulness dan Answer Relevance
  │
  ├── Metric Calculation
  │   └── Agregasi skor per metrik:
  │       • Hit Rate = ( jumlah query dengan hit ) / total query
  │       • MRR = Σ (1 / rank_relevan) / total query
  │       • Faithfulness = klaim_valid / total_klaim
  │       • Answer Relevance = mean(cosine_similarity)
  │
  └── Comparative Analysis
      └── Bandingkan performa:
          • Baseline: Fixed-size Chunking + BM25
          • Usulan: Semantic Chunking + Hybrid Retrieval + RRF

Phase 4: Output Generation
  │
  ├── Individual Results
  │   └── Per kueri: query, retrieved_chunks, generated_answer,
  │       ground_truth, hit_rate_status, mrr_score,
  │       faithfulness_score, answer_relevance_score
  │
  ├── Aggregate Statistics
  │   └── Rata-rata, standar deviasi, minimum, maximum per metrik
  │
  └── Export Formats
      ├── CSV: hasil_akhir_ragas_skripsi.csv
      ├── Excel: Laporan terperinci dengan pivot tables
      └── JSON: Untuk integrasi dengan dashboard visualisasi
```

### 4.3 Ground Truth Integration Logic

```python
# Pseudocode: Ground Truth Integration dalam Pipeline Evaluasi

PROCEDURE EvaluateWithGoldenDataset(golden_dataset_path, system_config):
    
    // Load Golden Dataset
    dataset ← LoadCSV(golden_dataset_path)
    results ← []
    
    FOR each row IN dataset:
        query ← row.Query
        expected_pasal ← row.Konteks_Pasal
        ground_truth ← row.Ground_Truth
        
        // Jalankan pipeline RAG
        retrieved_chunks ← HybridRetrieval(query, system_config)
        generated_answer ← LLMGenerate(query, retrieved_chunks)
        
        // Evaluasi Retrieval
        hit ← CheckHit(retrieved_chunks, expected_pasal)
        rank ← GetFirstRelevantRank(retrieved_chunks, expected_pasal)
        
        // Evaluasi Generative (RAGAS)
        faithfulness ← CalculateFaithfulness(generated_answer, retrieved_chunks)
        answer_relevance ← CalculateAnswerRelevance(generated_answer, query)
        
        // Simpan hasil
        results.append({
            id: row.ID,
            query: query,
            expected_pasal: expected_pasal,
            ground_truth: ground_truth,
            hit: hit,
            rank: rank,
            faithfulness: faithfulness,
            answer_relevance: answer_relevance,
            generated_answer: generated_answer
        })
    
    // Agregasi metrik
    hit_rate ← Mean([r.hit FOR r IN results])
    mrr ← Mean([1/r.rank FOR r IN results IF r.hit == 1])
    avg_faithfulness ← Mean([r.faithfulness FOR r IN results])
    avg_answer_relevance ← Mean([r.answer_relevance FOR r IN results])
    
    RETURN {
        individual_results: results,
        aggregated_metrics: {
            hit_rate: hit_rate,
            mrr: mrr,
            avg_faithfulness: avg_faithfulness,
            avg_answer_relevance: avg_answer_relevance
        }
    }
```

### 4.4 Expert Verification Protocol

```
PROTOCOL VERIFIKASI AHLI HUKUM
═══════════════════════════════════════════════════════════════

Tahap 1: Validasi Isi
  ├── Penelaahan setiap skenario kasus oleh ahli hukum
  ├── Verifikasi ketepatan referensi pasal terhadap KUHP Baru
  ├── Validasi kesesuaian ground truth dengan teks regulasi
  └── Penandaan skenario yang memerlukan revisi

Tahap 2: Validasi Kelengkapan
  ├── Pemeriksaan cakupan topik hukum:
  │   ├── Asas umum (Pasal 1-6)
  │   ├── Unsur tindak pidana (Pasal 10-19)
  │   ├── Alasan penghapus pidana (Pasal 31-44)
  │   ├── Korporasi sebagai subjek hukum (Pasal 45-49)
  │   ├── Tujuan pemidanaan (Pasal 51-53)
  │   └── Jenis pidana (Pasal 65-100)
  ├── Identifikasi gap cakupan
  └── Rekomendasi penambahan skenario jika diperlukan

Tahap 3: Validasi Kelayakan Evaluasi
  ├── Penilaian kesesuaian skenario dengan tujuan penelitian
  ├── Verifikasi bahwa skenario menguji konteks lintas pasal
  └── Persetujuan final untuk penggunaan dalam evaluasi

Tahap 4: Dokumentasi
  ├── Berita acara verifikasi ahli
  ├── Daftar skenario yang direvisi
  └── Sertifikat kelayakan dataset
```

### 4.5 Data Quality Assurance

| Aspek | Kriteria | Metode Verifikasi |
|-------|----------|-------------------|
| **Kesesuaian Referensi** | Setiap ground truth memiliki referensi pasal yang valid | Cross-check dengan kuhp_bersih.json |
| **Kecukupan Jawaban** | Ground truth menjawab secara lengkap query pengguna | Validasi oleh ahli hukum |
| **Konsistensi** | Tidak ada kontradiksi antar skenario dalam dataset | Automated consistency check |
| **Cakupan** | Mencakup minimal 5 topik hukum utama | Topic modeling analysis |
| **Kebersihan Teks** | Tidak ada artefak preprocessing dalam query dan ground truth | Regex validation |

### 4.6 OCR Data Quality Limitations

Sumber data primer yaitu file PDF `KUHP BARU UU Nomor 1 Tahun 2023` memiliki keterbatasan kualitas ekstraksi teks yang perlu didokumentasikan:

| Aspek | Kondisi | Dampak | Mitigasi |
|-------|---------|--------|----------|
| **Kualitas OCR** | Tidak merata antar halaman | Beberapa pasal awal (1, 2, 3, 5, 6) diekstrak sebagai teks pendek atau `"Cukup jelas"` | Dokumentasi eksplisit; tidak digunakan sebagai ground truth untuk evaluasi kuantitatif |
| **Artefak OCR** | Karakter rusak: `FRESIDEN`, `REPUBUK`, `INOONESIA`, `Ayatl2l`, dll. | Dapat mengganggu proses cleaning dan chunking | Module `OcrCleaner` dengan 50+ regex replacement rules |
| **Konsistensi Metadata** | Tidak semua chunk memiliki referensi pasal yang eksplisit | Evaluasi otomatis mungkin miss relevant chunks | Sistem menggunakan **backup pasal-level corpus** untuk segmentasi eksplisit sebelum semantic chunking |
| **Keseimbangan Retrieval** | Query tentang pasal dengan ekstraksi buruk akan mendapatkan hasil yang kurang relevan | Hit Rate untuk pasal tersebut menjadi rendah | Diakui sebagai **baseline perbandingan** di Bab 4; bukan kegagalan sistem |

**Catatan Penting untuk Evaluasi:**
- Golden Dataset hanya menggunakan pasal dengan ekstraksi teks yang cukup baik untuk ground truth yang dapat diverifikasi.
- Metrik evaluasi (Hit Rate, MRR) dihitung dengan **pasal mapping** yang di-generate dari corpus bersih, bukan dari nomor pasal secara harfiah.
- Hasil evaluasi yang rendah pada pasal tertentu tidak necessarily mengindikasikan kegagalan sistem, melainkan keterbatasan data sumber.

### 4.7 Hallucination Mitigation Strategy

Sistem menerapkan strategi berlapis untuk mencegah LLM menghasilkan jawaban di luar konteks regulasi:

| Layer | Mekanisme | Implementasi |
|-------|-----------|--------------|
| **1. Prompt Engineering** | System prompt dengan instruksi refusal eksplisit | `LEGAL_QA_TEMPLATE` memaksa LLM mengembalikan kalimat standar jika konteks tidak relevan |
| **2. Retrieval Filtering (CRAG)** | Threshold skor minimum untuk menyaring dokumen tidak relevan | `ReRanker.filter()` dengan threshold `0.010` |
| **3. Empty Context Handling** | UI menampilkan pesan standar jika 0 hasil melewati threshold | `render_no_context_message()` |
| **4. Post-generation Validation** | Deteksi refusal pattern di output LLM | `LLMEngine.is_refusal()` dengan regex patterns |
| **5. Local LLM Deployment** | Tidak ada API eksternal; data tidak keluar dari sistem | Ollama + Llama-3 8B running locally |

**Prompt Refusal Pattern:**
```
Jika DOKUMEN KONTEKS tidak memuat informasi yang relevan untuk menjawab 
PERTANYAAN PENGGUNA, jawab dengan persis kalimat berikut:
"Saya tidak dapat menemukan pasal yang relevan dalam KUHP Baru untuk 
menjawab pertanyaan ini."
```

**Audit Trail:**
- Semua refusal (detected via `is_refusal()`) dicatat untuk analisis selanjutnya.
- Refusal rate menjadi metrik tambahan dalam evaluasi sistem.

---

## 5. SYSTEM CONFIGURATION REFERENCE

### 5.1 Default Configuration Parameters

```yaml
system_config:
  preprocessing:
    document_cleaning:
      remove_page_numbers: true
      merge_broken_sentences: true
      preserve_article_markers: ["Pasal", "Ayat", "huruf"]
    
    normalization:
      lowercase: true
      expand_abbreviations:
        "uu": "undang-undang"
        "kuhp": "kitab undang-undang hukum pidana"
    
    semantic_chunking:
      enabled: true
      threshold: 0.75
      embedding_model: "BAAI/bge-m3"
      sentence_splitter: "indonesian-nlp"
  
  retrieval:
    hybrid_search:
      enabled: true
      sparse:
        algorithm: "BM25"
        k1: 1.5
        b: 0.75
        top_k: 10
      dense:
        algorithm: "CosineSimilarity"
      embedding_model: "indobenchmark/indobert-base-p1"
        top_k: 10
    
    rrf:
      enabled: true
      k: 60
      top_k_final: 5
  
  generation:
    llm:
      model: "meta-llama/Meta-Llama-3-8B-Instruct"
      quantization: "4-bit (GGUF/AWQ)"
      max_tokens: 512
      temperature: 0.1
      top_p: 0.9
    
    prompt_template: |
      Anda adalah asisten hukum AI yang hanya menjawab berdasarkan dokumen konteks.
      
      DOKUMEN KONTEKS:
      {context}
      
      PERTANYAAN: {query}
      
      Instruksi:
      1. Jawab hanya berdasarkan dokumen konteks yang diberikan.
      2. Sertakan sitasi pasal yang relevan.
      3. Jika jawaban tidak ada di konteks, katakan "Saya tidak tahu. Informasi tersebut tidak ditemukan dalam regulasi yang tersedia."
      4. Jangan menambahkan informasi di luar dokumen konteks.
      
      JAWABAN:
  
  evaluation:
    metrics:
      - "Hit Rate"
      - "MRR"
      - "Faithfulness"
      - "Answer Relevance"
    golden_dataset_path: "golden_dataset_rag_hukum_indonesia_rev3.csv"
    batch_size: 50
  
  infrastructure:
    hardware: "Asus Vivobook Pro 14 OLED"
    hpc_option: "BRIN (opsional untuk tahap pengujian)"
    vector_db: "FAISS/Chroma"
```

---

## 6. PERFORMANCE BENCHMARKS

### 6.1 Target Metrics

| Metrik | Baseline Target | Usulan Target | Satuan |
|--------|-----------------|---------------|--------|
| Hit Rate | ≥ 0.70 | ≥ 0.85 | Ratio |
| MRR | ≥ 0.65 | ≥ 0.80 | Ratio |
| Faithfulness | ≥ 0.75 | ≥ 0.90 | Ratio |
| Answer Relevance | ≥ 0.70 | ≥ 0.85 | Ratio |

### 6.2 Computational Constraints

| Aspek | Spesifikasi | Catatan |
|-------|-------------|---------|
| RAM | 16 GB | Minimum untuk inferensi 4-bit |
| VRAM | 8 GB | Untuk model Llama-3 8B kuantisasi 4-bit |
| Storage | 50 GB | Korpus regulasi + vector database + model |
| Inference Time | < 30 detik | Per query (termasuk retrieval dan generation) |

---

## 7. DELIVERABLES

### 7.1 Software Artifacts

| ID | Nama | Format | Deskripsi |
|----|------|--------|-----------|
| D-01 | Aplikasi Streamlit | Python package | Antarmuka web untuk interaksi pengguna |
| D-02 | Modul Preprocessing | Python module | Document cleaning, normalization, semantic chunking |
| D-03 | Modul Retrieval | Python module | BM25, Dense, Hybrid, RRF |
| D-04 | Modul Generation | Python module | LLM inference dengan prompt engineering |
| D-05 | Modul Evaluasi | Python module | Hit Rate, MRR, RAGAS metrics |
| D-06 | Vector Database | FAISS/Chroma files | Indexed korpus KUHP Baru |
| D-07 | Golden Dataset | CSV/Excel | 50+ skenario kasus dengan ground truth |

### 7.2 Documentation Artifacts

| ID | Nama | Format | Deskripsi |
|----|------|--------|-----------|
| D-08 | Technical Specification | Markdown | Dokumen ini |
| D-09 | User Manual | PDF/Markdown | Panduan penggunaan untuk praktisi hukum |
| D-10 | API Documentation | Markdown/OpenAPI | Dokumentasi endpoint jika ada REST API |
| D-11 | Evaluation Report | CSV/Excel | Hasil pengujian kuantitatif |
| D-12 | Thesis Chapters IV & V | PDF | Laporan hasil dan kesimpulan penelitian |

---

## 8. APPENDIX

### 8.1 Formula Reference

**BM25 Score:**
```
score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D| / avgdl))
```

**Cosine Similarity:**
```
cos(θ) = (A · B) / (||A|| × ||B||) = Σ(Ai × Bi) / √(ΣAi²) × √(ΣBi²)
```

**RRF Score:**
```
RRF(d) = Σ 1 / (k + rank_i(d)) untuk setiap sistem pencarian i
```

**Hit Rate:**
```
HitRate@K = (jumlah query dengan minimal 1 dokumen relevan di Top-K) / total query
```

**MRR:**
```
MRR = (1/|Q|) × Σ (1 / rank_relevan_pertama(q))
```

**Faithfulness:**
```
Faithfulness = (jumlah klaim yang diverifikasi valid) / (total klaim dalam jawaban)
```

**Answer Relevance:**
```
AnswerRelevance = mean(CosineSimilarity(embed(query_asli), embed(query_buatan_i)))
```

### 8.2 Glossary

| Term | Deskripsi |
|------|-----------|
| **RAG** | Retrieval-Augmented Generation - arsitektur LLM dengan basis pengetahuan eksternal |
| **Semantic Chunking** | Metode pemotongan dokumen berdasarkan kesatuan makna semantik |
| **Hybrid Retrieval** | Kombinasi sparse (BM25) dan dense retrieval |
| **RRF** | Reciprocal Rank Fusion - algoritma re-ranking hasil pencarian gabungan |
| **Faithfulness** | Tingkat kesetiaan jawaban terhadap konteks dokumen |
| **Answer Relevance** | Tingkat kesesuaian jawaban dengan pertanyaan pengguna |
| **Hallucination** | Fenomena LLM menghasilkan informasi yang tidak ada di konteks |
| **Golden Dataset** | Dataset evaluasi dengan ground truth yang telah diverifikasi ahli |
| **Human-Centered AI** | Prinsip penempatan manusia sebagai pengambil keputusan utama |
| **Vector Database** | Basis data untuk penyimpanan dan pencarian representasi vektor |

---

## 9. REFERENSI DOKUMEN

- Proposal Tugas Akhir: `docs/proposal/152023007_File_Proposal_Rainova_Rahaniawan_rev2.docx`
- Korpus Regulasi: `../data/raw/KUHP BARU UU Nomor 1 Tahun 2023.pdf`
- Dataset Evaluasi: `../data/datasets/golden_dataset_rag_hukum_indonesia_rev3.csv`
- Korpus Bersih: `../data/processed/kuhp_bersih.json`
- Diagram Alur Sistem: `docs/diagrams/Diagram Alur Sistem Retrieval-Augmented Generation (RAG) pada Domain Regulasi.drawio`
- Wireframe UI: `docs/diagrams/Wireframe Website.fig`
- Coding Standards: `STYLE_GUIDE.md`
- Tech Stack: `TECH_STACK.md`
- Architecture Assessment: `ASSESSMENT.md`
- Database Schema: `database.sql`

---

**Dokumen ini disusun untuk mendukung implementasi sistem RAG Asisten Regulasi**
**sebagaimana diuraikan dalam Proposal Tugas Akhir Semester 7.**
