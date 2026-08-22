import json
import os
import time
import warnings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Mengabaikan warning deprecation dari langchain agar terminal lebih bersih
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    print("="*60)
    print("FASE 2.2: PEMBANGUNAN FAISS VECTOR DB DARI KUHP BERSIH")
    print("="*60)

    # 1. Load Data Chunk 
    input_file = "kuhp_bersih.json"
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' tidak ditemukan!")
        return

    print(f"🔄 Membaca berkas '{input_file}'...")
    with open(input_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    # 2. Konversi JSON ke Format Document LangChain (Deteksi Otomatis)
    documents = []
    
    if isinstance(chunks_data, list):
        for i, item in enumerate(chunks_data):
            if isinstance(item, dict):
                # Jika formatnya dictionary (punya key text, pasal, dll)
                page_content = item.get("text", "") or item.get("page_content", "")
                metadata = {
                    "pasal": item.get("pasal", "N/A"),
                    "bab": item.get("bab", "N/A"),
                    "buku": item.get("buku", "N/A"),
                    "sumber": "KUHP BARU UU No 1 Tahun 2023"
                }
            elif isinstance(item, str):
                # Jika formatnya hanya list of strings murni (seperti kasus Anda saat ini)
                page_content = item
                metadata = {
                    "chunk_id": f"chunk_{i+1}",
                    "sumber": "KUHP BARU UU No 1 Tahun 2023"
                }
            else:
                continue
            
            # Hanya masukkan teks yang tidak kosong
            if page_content.strip(): 
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)
                
    elif isinstance(chunks_data, dict):
         # Berjaga-jaga jika format JSON-nya dictionary bersarang
         for key, val in chunks_data.items():
             page_content = str(val)
             metadata = {"chunk_id": key, "sumber": "KUHP BARU UU No 1 Tahun 2023"}
             documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"✅ Berhasil mengekstrak dan menyiapkan {len(documents)} dokumen chunks.")

    # 3. Inisialisasi Model Embedding Multilingual
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"\n🔄 Mengunduh / Memuat Embedding Model: '{model_name}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'} # Ganti ke 'cuda' jika Anda pakai GPU
    )

    # 4. Membangun Vektor DB FAISS
    print("\n⚡ Memproses vektorisasi teks ke dalam indeks FAISS... (Mohon tunggu, butuh beberapa saat)")
    start_time = time.time()
    
    vector_store = FAISS.from_documents(documents, embeddings)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Indeks FAISS berhasil dibuat dalam waktu {elapsed_time:.2f} detik.")

    # 5. Menyimpan Indeks FAISS ke Disk Lokal
    output_dir = "faiss_index_kuhp"
    print(f"\n💾 Menyimpan direktori FAISS ke '{output_dir}'...")
    vector_store.save_local(output_dir)
    print(f"🎉 Selesai! Indeks FAISS tersimpan dengan aman.")

    # 6. Pengujian Sederhana (Sanity Check)
    print("\n" + "="*60)
    print("SANITY CHECK: Pengujian Pencarian Semantik Sederhana")
    print("="*60)
    query_test = "pencurian dengan pemberatan atau pembongkaran"
    print(f"🔍 Kueri Uji: '{query_test}'")
    
    results = vector_store.similarity_search_with_score(query_test, k=2)
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Hasil #{i} (Distance Score: {score:.4f}) ---")
        print(f"📌 Metadata: {doc.metadata}")
        print(f"📜 Teks Chunk:\n{doc.page_content[:250]}...")

if __name__ == "__main__":
    main()