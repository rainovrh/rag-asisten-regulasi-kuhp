import os
import json
import warnings
import pandas as pd
import numpy as np
from time import time
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

warnings.filterwarnings("ignore")

def main():
    print("="*70)
    print("FASE 4: EKSEKUSI MASSAL LLAMA-3 (50 GOLDEN DATASET)")
    print("="*70)

    # ---------------------------------------------------------
    # 1. SETUP RETRIEVER (BM25 + FAISS + CRAG)
    # ---------------------------------------------------------
    print("[1/4] Menyiapkan Mesin Pencari Hybrid & CRAG...")
    with open("kuhp_bersih.json", 'r', encoding='utf-8') as f:
        kuhp_data = json.load(f)

    pasal_ids = list(kuhp_data.keys())
    pasal_texts = list(kuhp_data.values())
    tokenized_corpus = [str(doc).lower().split() for doc in pasal_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.load_local("faiss_index_kuhp", embeddings, allow_dangerous_deserialization=True)

    def retrieve_context(query, top_n=3, crag_threshold=0.015):
        # A. Leksikal
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:10]
        bm25_ranks = [pasal_ids[i] for i in bm25_top_indices]
        
        # B. Semantik
        faiss_results = vectorstore.similarity_search(query, k=10)
        faiss_ranks = [doc.metadata.get('chunk_id', doc.metadata.get('pasal', '')) for doc in faiss_results]
        
        # C. RRF
        rrf_scores = {}
        k_rrf = 60
        for rank, doc_id in enumerate(bm25_ranks):
            if doc_id: rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
        for rank, doc_id in enumerate(faiss_ranks):
            if doc_id: rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
            
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # D. CRAG Filter & Formatting Context
        final_context_texts = []
        for doc_id, score in sorted_rrf:
            if score >= crag_threshold:
                pasal_text = kuhp_data.get(doc_id, "")
                final_context_texts.append(f"[{doc_id}]: {pasal_text}")
            if len(final_context_texts) == top_n:
                break
                
        return "\n\n".join(final_context_texts)

    # ---------------------------------------------------------
    # 2. SETUP LLM & SYSTEM PROMPT (Sesuai Langkah 2.4)
    # ---------------------------------------------------------
    print("[2/4] Menghubungkan ke LLM Lokal (Llama-3 via Ollama)...")
    llm = Ollama(model="llama3")

    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template="""Anda adalah asisten hukum AI. Jawablah murni berdasarkan DOKUMEN KONTEKS yang diberikan di bawah ini. 
Jika jawaban tidak ada di dalam dokumen konteks, katakan 'Saya tidak tahu' atau 'Informasi tidak tersedia di konteks' untuk menghindari halusinasi. Jangan membuat karangan. Gunakan Bahasa Indonesia yang formal dan selalu sebutkan dasar Pasalnya.

DOKUMEN KONTEKS:
{context}

PERTANYAAN PENGGUNA:
{query}

JAWABAN:
"""
    )

    # ---------------------------------------------------------
    # 3. PROSES DATASET (Prioritas 1)
    # ---------------------------------------------------------
    dataset_path = "golden_dataset_rag_hukum_indonesia_rev3.csv"
    print(f"[3/4] Membaca dataset dari '{dataset_path}'...")
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"❌ Gagal membaca dataset: {e}")
        return

    # Menyiapkan kolom baru untuk menyimpan hasil eksekusi
    df['Retrieved_Context'] = ""
    df['AI_Answer'] = ""

    print(f"[4/4] 🚀 Memulai Looping {len(df)} Kueri. Ini akan memakan waktu, mohon tunggu...")
    start_time = time()

    for index, row in df.iterrows():
        query = str(row['Query'])
        print(f"\n⏳ Proses [{index+1}/{len(df)}]: {query[:50]}...")
        
        # 1. Tarik Konteks Hukum
        context = retrieve_context(query)
        df.at[index, 'Retrieved_Context'] = context
        
        # 2. Jika tidak ada konteks (Dibuang oleh CRAG semua), bypass LLM untuk cegah halusinasi
        if not context.strip():
            ai_answer = "Informasi tidak tersedia di konteks. (Dicegah oleh filter CRAG)"
        else:
            # 3. Generate Jawaban LLM
            prompt_ready = prompt_template.format(context=context, query=query)
            try:
                ai_answer = llm.invoke(prompt_ready)
            except Exception as e:
                ai_answer = f"Error LLM: {str(e)}"
        
        df.at[index, 'AI_Answer'] = ai_answer
        print(f"   ✅ Selesai. (Panjang Jawaban: {len(ai_answer)} karakter)")

    # ---------------------------------------------------------
    # 4. SIMPAN HASIL KE CSV BARU
    # ---------------------------------------------------------
    output_path = "hasil_generasi_llama3.csv"
    df.to_csv(output_path, index=False)
    
    elapsed = (time() - start_time) / 60
    print("\n" + "="*70)
    print(f"🎉 EKSEKUSI MASSAL SELESAI DALAM {elapsed:.2f} MENIT!")
    print(f"💾 File hasil evaluasi disimpan di: '{output_path}'")
    print("File ini sudah siap digunakan untuk perhitungan metrik evaluasi RAGAS (Fase 5).")
    print("="*70)

if __name__ == "__main__":
    main()