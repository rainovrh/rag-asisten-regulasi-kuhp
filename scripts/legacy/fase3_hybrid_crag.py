import json
import warnings
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Mengabaikan warning deprecation dari langchain agar terminal lebih bersih
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    print("="*70)
    print("FASE 3: IMPLEMENTASI HYBRID RETRIEVAL, RRF, & FILTER CRAG")
    print("="*70)

    # ==========================================
    # 1. PERSIAPAN DATA & MODEL LEKSIKAL (BM25)
    # ==========================================
    print("[1/4] Memuat dokumen KUHP dan Indeks Leksikal (BM25)...")
    with open("kuhp_bersih.json", 'r', encoding='utf-8') as f:
        kuhp_data = json.load(f)

    # Berdasarkan output fase1_cleaning.py Anda, data berbentuk dictionary (Key: Pasal, Value: Teks)
    pasal_ids = list(kuhp_data.keys())
    pasal_texts = list(kuhp_data.values())
    
    # Tokenisasi untuk BM25
    tokenized_corpus = [str(doc).lower().split() for doc in pasal_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # ==========================================
    # 2. PERSIAPAN MODEL SEMANTIK (FAISS)
    # ==========================================
    print("[2/4] Memuat Indeks Semantik (FAISS)...")
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    
    # Memuat indeks FAISS lokal (allow_dangerous_deserialization diperlukan untuk versi Langchain terbaru)
    vectorstore = FAISS.load_local("faiss_index_kuhp", embeddings, allow_dangerous_deserialization=True)

    # ==========================================
    # 3. FUNGSI RECIPROCAL RANK FUSION (RRF)
    # ==========================================
    def reciprocal_rank_fusion(bm25_ranks, faiss_ranks, k=60):
        rrf_scores = {}
        # Hitung skor untuk BM25
        for rank, doc_id in enumerate(bm25_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            
        # Hitung skor untuk FAISS
        for rank, doc_id in enumerate(faiss_ranks):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            
        # Urutkan berdasarkan skor RRF tertinggi
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # ==========================================
    # 4. FUNGSI PENCARIAN HYBRID + CRAG FILTER
    # ==========================================
    def hybrid_search_with_crag(query, top_n=3, crag_threshold=0.015):
        print(f"\n[3/4] MENJALANKAN HYBRID SEARCH & CRAG")
        print(f"Kueri: '{query}'\n")
        
        # A. Pencarian Leksikal (BM25) - Ambil Top 10
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:10]
        bm25_ranks = [pasal_ids[i] for i in bm25_top_indices]
        
        # B. Pencarian Semantik (FAISS) - Ambil Top 10
        faiss_results = vectorstore.similarity_search(query, k=10)
        faiss_ranks = []
        for doc in faiss_results:
            # Mengambil ID Pasal dari metadata FAISS yang dibuat di Fase 2
            doc_id = doc.metadata.get('chunk_id') or doc.metadata.get('pasal')
            if doc_id:
                faiss_ranks.append(doc_id)
                
        # C. Gabungkan dengan RRF
        rrf_results = reciprocal_rank_fusion(bm25_ranks, faiss_ranks)
        
        # D. Logika CRAG (Corrective RAG - Filtering)
        print("[4/4] 🛡️ Menjalankan Filter CRAG (Membuang konteks sampah)...")
        final_context = []
        dropped_context = 0
        
        for doc_id, score in rrf_results:
            # Jika skor RRF di bawah ambang batas (Threshold), anggap irelevan dan BUANG.
            if score >= crag_threshold:
                final_context.append((doc_id, score))
            else:
                dropped_context += 1
                
            # Berhenti jika sudah mengumpulkan n dokumen yang valid
            if len(final_context) == top_n:
                break
                
        # --- MENAMPILKAN HASIL ---
        print("\n" + "="*50)
        print(f"🏆 HASIL FINAL RRF & CRAG (TOP {top_n} PASAL RELEVAN):")
        print("="*50)
        
        if not final_context:
            print("❌ Tidak ada pasal yang cukup relevan (Semua di-drop oleh CRAG). Sistem akan mencegah LLM berhalusinasi.")
        
        for i, (doc_id, score) in enumerate(final_context):
            print(f"{i+1}. {doc_id} (Skor RRF: {score:.4f} | Status: LOLOS CRAG)")
            snippet = str(kuhp_data.get(doc_id, ""))[:200].replace('\n', ' ') + "..."
            print(f"   📜 Teks: {snippet}\n")
            
        if dropped_context > 0:
            print(f"🗑️ INFO CRAG: {dropped_context} dokumen dibuang karena dinilai tidak relevan (Skor < {crag_threshold}).")

        return final_context

    # --- UJI COBA ---
    test_query = "Seseorang menggasak barang milik orang lain secara diam-diam di malam hari"
    hybrid_search_with_crag(test_query, top_n=3)

if __name__ == "__main__":
    main()