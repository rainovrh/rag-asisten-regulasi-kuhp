import streamlit as st
import json
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Asisten Hukum AI (RAG KUHP)", page_icon="⚖️", layout="wide")
st.title("⚖️ Asisten Hukum Berbasis AI (KUHP Baru)")
st.caption("Human-Centered AI: Jawaban dihasilkan berbasis rujukan pasal resmi KUHP (UU No. 1 Tahun 2023).")

# ==========================================
# 2. CACHING MODEL & DATA
# ==========================================
@st.cache_resource
def load_system():
    with open("kuhp_bersih.json", 'r', encoding='utf-8') as f:
        kuhp_data = json.load(f)
    pasal_ids = list(kuhp_data.keys())
    pasal_texts = list(kuhp_data.values())
    
    tokenized_corpus = [str(doc).lower().split() for doc in pasal_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.load_local("faiss_index_kuhp", embeddings, allow_dangerous_deserialization=True)
    
    llm = Ollama(model="llama3", temperature=0.0) # Temperature=0 agar output sangat konsisten & tidak berhalusinasi
    
    return kuhp_data, pasal_ids, bm25, vectorstore, llm

with st.spinner("Memuat Basis Pengetahuan KUHP dan Model AI..."):
    kuhp_data, pasal_ids, bm25, vectorstore, llm = load_system()

# ==========================================
# 3. FUNGSI RETRIEVAL (HYBRID + RRF + CRAG DENGAN THRESHOLD KETAT)
# ==========================================
def retrieve_context(query, top_n=3, crag_threshold=0.020): # Threshold dinaikkan ke 0.020 agar pasal irelevan di-drop
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:10]
    bm25_ranks = [pasal_ids[i] for i in bm25_top_indices]
    
    faiss_results = vectorstore.similarity_search(query, k=10)
    faiss_ranks = [doc.metadata.get('chunk_id', doc.metadata.get('pasal', '')) for doc in faiss_results]
    
    rrf_scores = {}
    k_rrf = 60
    for rank, doc_id in enumerate(bm25_ranks):
        if doc_id: rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
    for rank, doc_id in enumerate(faiss_ranks):
        if doc_id: rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
        
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_context_list = []
    for doc_id, score in sorted_rrf:
        if score >= crag_threshold: # Hanya pasal dengan skor >= 0.020 yang lolos
            pasal_text = kuhp_data.get(doc_id, "")
            final_context_list.append({"pasal": doc_id, "teks": pasal_text, "skor": score})
        if len(final_context_list) == top_n:
            break
            
    return final_context_list

# ==========================================
# 4. SYSTEM PROMPT KETAT (TANPA BASA-BASI)
# ==========================================
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="""Anda adalah pakar hukum AI yang sangat ketat dan objektif. 
Jawablah pertanyaan berikut HANYA berdasarkan DOKUMEN KONTEKS di bawah ini.
DILARANG memberikan asumsi, pendapat pribadi, atau saran di luar isi dokumen konteks.
Jelaskan unsur pidana atau sanksinya secara ringkas, formal, dan sebutkan nomor Pasalnya secara jelas.

DOKUMEN KONTEKS:
{context}

PERTANYAAN PENGGUNA:
{query}

JAWABAN RELEVAN:
"""
)

# ==========================================
# 5. UI CHAT STREAMLIT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "context_used" in msg and msg["context_used"]:
            with st.expander("🔍 Lihat Referensi Pasal Terverifikasi (Traceability)"):
                for ctx in msg["context_used"]:
                    st.info(f"**{ctx['pasal']}** (Skor Keandalan RRF: {ctx['skor']:.4f})\n\n{ctx['teks']}")

if user_query := st.chat_input("Tanyakan permasalahan hukum di sini..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Mencari pasal KUHP dan Menganalisis..."):
            retrieved_data = retrieve_context(user_query)
            
            # LOGIKA STRICT: Jika CRAG membuang semua pasal, BYPASS LLM!
            if not retrieved_data:
                ai_response = "⚠️ **Informasi Tidak Ditemukan dalam KUHP Baru.**\n\nSistem (Filter CRAG) membuang konteks karena tidak ditemukan pasal yang memiliki tingkat relevansi memadai dengan kata kunci Anda. Untuk menjaga keandalan hukum dan mencegah halusinasi AI, analisis tidak diberikan."
            else:
                context_str = "\n\n".join([f"[{item['pasal']}]: {item['teks']}" for item in retrieved_data])
                prompt_ready = prompt_template.format(context=context_str, query=user_query)
                ai_response = llm.invoke(prompt_ready)
            
            st.markdown(ai_response)
            
            if retrieved_data:
                with st.expander("🔍 Lihat Referensi Pasal Terverifikasi (Traceability)"):
                    for ctx in retrieved_data:
                        st.info(f"**{ctx['pasal']}** (Skor Keandalan RRF: {ctx['skor']:.4f})\n\n{ctx['teks']}")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": ai_response,
                "context_used": retrieved_data
            })