# Human-in-the-Loop Evaluation Template
## Evaluasi Kualitatif Jawaban AI untuk Skripsi

### 1. Tujuan Evaluasi

Menilai kewarasan nalar hukum dari jawaban yang dihasilkan sistem RAG Asisten Regulasi untuk KUHP Baru (UU No. 1 Tahun 2023). Evaluasi ini mengambil sampel **15-20% dari total 50 skenario** (≈ **7-10 kasus**) untuk dievaluasi secara manual oleh peneliti.

### 2. Metodologi Sampling

- **Populasi**: 50 skenario evaluasi golden dataset
- **Sampel**: 7-10 skenario (15-20%)
- **Teknik Sampling**: *Purposive sampling* — pilih skenario dengan karakteristik:
  - Campuran factoid dan open-ended
  - Rentang pasal yang berbeda (awal, tengah, akhir)
  - Variasi tingkat kesulitan

### 3. Kriteria Evaluasi

Setiap jawaban AI dievaluasi berdasarkan 3 kriteria utama:

| # | Kriteria | Bobot | Deskripsi |
|---|----------|-------|-----------|
| 1 | **Akurasi Hukum** | 40% | Sesuai dengan teks KUHP Baru |
| 2 | **Kelengkapan Situsasi** | 30% | Menyebut nomor pasal/ayat secara benar |
| 3 | **Konsistensi Logika** | 30% | Tidak ada kontradiksi internal |

### 4. Skala Penilaian

- **5** = Sangat Baik (sempurna)
- **4** = Baik (minor issues)
- **3** = Cukup (masih dapat diterima)
- **2** = Kurang (ada kesalahan signifikan)
- **1** = Buruk (salah secara materiil)

### 5. Template Evaluasi

```
Evaluasi Manual - Skenario #[ID]
=====================================

Query: [pertanyaan]
Ground Truth: [jawaban ideal]
AI Answer: [jawaban sistem]
Pasal Referensi: [pasal yang relevan]

Penilaian:
1. Akurasi Hukum: [1-5]
   Catatan: [...]
   
2. Kelengkapan Situsasi: [1-5]
   Catatan: [...]
   
3. Konsistensi Logika: [1-5]
   Catatan: [...]

Skor Total: (1x0.4) + (2x0.3) + (3x0.3) = [nilai]

Kesimpulan: [Accept/Reject/Revise]
```

### 6. Validasi Silang Matematis

Setiap skenario yang dievaluasi harus melalui verifikasi matematis:

**Hit Rate Verification:**
```
Hit = 1 jika doc_id yang diharapkan muncul di top-5 retrieval, else 0
Hit Rate@5 = Sigma(Hit) / N
```

**MRR Verification:**
```
RR = 1 / rank_pertama_doc_relevan  (0 jika tidak ditemukan)
MRR = rata-rata(RR) untuk semua query
```

**Contoh Perhitungan Manual:**
```
Query 1: [query], Expected: pasal 34
Retrieved: [pasal 34_chunk_3, pasal 43_chunk_0, pasal 228_chunk_0, ...]
Rank pertama relevan: 1
RR = 1/1 = 1.0
Hit = 1

Query 2: [query], Expected: pasal 56
Retrieved: [pasal 12_chunk_0, pasal 34_chunk_2, ...]
Rank pertama relevan: -1 (tidak ditemukan)
RR = 0
Hit = 0

MRR = (1.0 + 0) / 2 = 0.5
Hit Rate = 1/2 = 0.5
```

### 7. Dokumentasi Hasil

Hasil evaluasi kualitatif dan kuantitatif disimpan dalam format:

```
File: logs/human_evaluation_results.csv
Kolom: ID, Query, Scenario, Akurasi_Hukum, Kelengkapan_Situsasi, 
       Konsistensi_Logika, Skor_Total, Kesimpulan, Catatan
```

### 8. Acceptance Criteria

- **Minimal 70%** skenario sampel mendapat skor total ≥ 3.0
- **Minimal 80%** jawaban AI tidak mengandung halusinasi hukum
- Semua sitasi pasal yang disebutkan dapat diverifikasi di `kuhp_bersih.json`
