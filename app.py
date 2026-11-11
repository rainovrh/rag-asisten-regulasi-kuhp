"""Streamlit Web Application for Indonesian Criminal Code (KUHP Baru UU No. 1 Tahun 2023) RAG System.

Blok 5, Kotak 18:
- Interactive User Interface (Streamlit)
- Academic Evaluation Summary Sidebar
- Dual Index Retrieval & Groq Reasoning
- Transparent Verifiable Top-K Legal Reference Expander
- Strict Grounding Guardrail Badge & Explainability
"""

import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import INDEXES_DIR, PROCESSED_DATA_DIR
from src.generation.llm_builder import GroqLLMEngine
from src.preprocessing.normalizer import LegalQueryNormalizer, UserQueryValidator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievedLegalChunk
from src.retrieval.indobert_embedding import IndoBERTEmbedding


def init_ui():
    """Initialize page config and global CSS styles."""
    st.set_page_config(
        page_title="Asisten Hukum AI - KUHP Baru UU No. 1/2023",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1.05rem;
            color: #4b5563;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
        }
        .metric-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #64748b;
        }
        .metric-value {
            font-size: 1.3rem;
            font-weight: 700;
            color: #0f172a;
        }
        .chunk-box {
            background-color: #ffffff;
            border-left: 4px solid #3b82f6;
            border-radius: 4px;
            padding: 14px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .chunk-header {
            font-weight: 700;
            color: #1e40af;
            margin-bottom: 4px;
        }
        .chunk-breadcrumb {
            font-size: 0.8rem;
            color: #6b7280;
            margin-bottom: 8px;
        }
        .chunk-text {
            font-size: 0.92rem;
            line-height: 1.5;
            color: #1f2937;
        }
        .chunk-penjelasan {
            background-color: #fefce8;
            border: 1px dashed #ca8a04;
            border-radius: 4px;
            padding: 8px 12px;
            margin-top: 8px;
            font-size: 0.88rem;
            color: #854d0e;
        }
        .badge-grounded {
            background-color: #dcfce7;
            color: #166534;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        .badge-warning {
            background-color: #fef3c7;
            color: #92400e;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# CACHED SYSTEM LOADER
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat Basis Pengetahuan KUHP & Model AI...")
def load_rag_system():
    """Load persistent embedding model, indices, and retriever pipeline."""
    # IndoBERT Embedding
    embedding = IndoBERTEmbedding("models/indobert-base-p1")

    # BM25 Sparse Index
    bm25 = BM25Retriever.load_index(PROCESSED_DATA_DIR / "bm25_index.pkl")

    # FAISS Dense Index
    dense = DenseRetriever.load_from_dir(PROCESSED_DATA_DIR / "faiss_index_kuhp", embedding=embedding)

    # Hybrid Retriever with Weighted RRF & Gated Lexical Primacy (Skenario D)
    retriever = HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        rrf_k=60,
        candidate_k=30,
        w_bm25=2.0,
        w_dense=0.3,
        bm25_primacy_threshold=12.0,
        bm25_primacy_ratio=1.05,
    )

    # LLM Reasoning Engine (Groq Cloud API)
    llm_engine = GroqLLMEngine(model="qwen/qwen3.8-27b", temperature=0.1)

    # Boundary Validator & Normalizer
    validator = UserQueryValidator()
    normalizer = LegalQueryNormalizer()

    return retriever, llm_engine, validator, normalizer


# -----------------------------------------------------------------------------
# SIDEBAR: ABOUT & EVALUATION PERFORMANCE
# -----------------------------------------------------------------------------
def render_sidebar():
    """Render informative sidebar with Chapter IV ablation and performance benchmarks."""
    with st.sidebar:
        st.markdown("### ⚖️ Asisten Hukum RAG KUHP Baru")
        st.caption("Penelitian Tugas Akhir Teknik Informatika ITENAS")

        st.markdown("---")
        st.markdown("#### 📊 Performa Sistem (Skenario D Usulan)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-title">Hit Rate@5</div>
                    <div class="metric-value" style="color: #16a34a;">88.00%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-title">MRR</div>
                    <div class="metric-value" style="color: #16a34a;">0.7057</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-title">Relevance</div>
                    <div class="metric-value" style="color: #2563eb;">87.56%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-title">Citation Acc.</div>
                    <div class="metric-value" style="color: #2563eb;">100%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### 🏛️ Spesifikasi Arsitektur")
        st.markdown(
            """
            - **Basis Data**: UU No. 1 Tahun 2023 (624 Pasal, 643 Chunks)
            - **Tafsir Autentik**: 292 Penjelasan Pasal Tersinkronisasi
            - **Representasi**: IndoBERT Embedding (IndexFlatIP Cosine) + BM25 Lexical
            - **Fusi**: Weighted RRF ($w_{\\text{bm25}}=2.0, w_{\\text{dense}}=0.3$) + Gated Primacy
            - **LLM Engine**: Groq Cloud API (`qwen/qwen3.8-27b`)
            - **Guardrail**: Strict Regex Legal Grounding
            """
        )

        st.markdown("---")
        st.markdown("#### ⚠️ Disclaimer Akademis")
        st.caption(
            "Sistem ini merupakan purwarupa penelitian akademis Tugas Akhir di ITENAS Bandung. "
            "Jawaban dihasilkan secara otomatis berbasis penelusuran norma KUHP Baru (UU No. 1 Tahun 2023) "
            "dan **bukan merupakan nasihat hukum resmi**. Konsultasikan kasus konkret kepada advokat berlisensi."
        )

        st.markdown("---")
        st.caption("© 2026 Rainova Rahaniawan — ITENAS Bandung")


# -----------------------------------------------------------------------------
# MAIN APPLICATION INTERFACE
# -----------------------------------------------------------------------------
def main():
    """Main Streamlit execution loop."""
    init_ui()
    render_sidebar()

    st.markdown('<div class="main-title">⚖️ Asisten Penelusuran & Penalaran KUHP Baru</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Sistem Tanya-Jawab Berbasis Doktrin Hukum & Undang-Undang Nomor 1 Tahun 2023 (Tri-Blok Hierarchical RAG)</div>',
        unsafe_allow_html=True,
    )

    # Initialize components
    try:
        retriever, llm_engine, validator, normalizer = load_rag_system()
        st.success("🟢 Sistem Retrieval & Reasoning Aktif (Dual Index: BM25 + FAISS IndoBERT)", icon="✅")
    except Exception as e:
        st.error(f"Gagal memuat komponen sistem: {e}")
        st.stop()

    # Session state for message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Quick Case Suggestions
    st.markdown("##### 💡 Contoh Kueri Kasus Yuridis:")
    sample_col1, sample_col2 = st.columns(2)
    sample_queries = [
        "Apakah seseorang bisa dipidana atas perbuatan yang belum diatur undang-undang?",
        "Sanksi pidana menyatakan memiliki kekuatan gaib atau santet untuk menimbulkan penyakit",
        "Unsur-unsur tindak pidana pembunuhan berencana dan ancaman sanksinya",
        "Kapan suatu tindak pidana dianggap dilakukan oleh korporasi?",
    ]

    selected_sample = None
    with sample_col1:
        if st.button(f"📌 {sample_queries[0]}", key="btn_q1", use_container_width=True):
            selected_sample = sample_queries[0]
        if st.button(f"📌 {sample_queries[2]}", key="btn_q3", use_container_width=True):
            selected_sample = sample_queries[2]
    with sample_col2:
        if st.button(f"📌 {sample_queries[1]}", key="btn_q2", use_container_width=True):
            selected_sample = sample_queries[1]
        if st.button(f"📌 {sample_queries[3]}", key="btn_q4", use_container_width=True):
            selected_sample = sample_queries[3]

    st.markdown("---")

    # Render previous conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "retrieved_chunks" in msg and msg["retrieved_chunks"]:
                render_reference_expander(msg["retrieved_chunks"])

    # User Input
    user_query = st.chat_input("Ketik pertanyaan kasus atau rumusan pasal yang ingin dianalisis...")
    if selected_sample:
        user_query = selected_sample

    if user_query:
        # 1. Validation & Boundary Check (Kotak 8)
        validation = validator.validate(user_query)
        if not validation.is_valid:
            st.warning(f"⚠️ {validation.error_message}")
            return

        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Retrieval & Generation Pipeline (Kotak 9 s.d. 17)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Menelusuri pasal relevan & memverifikasi sitasi yuridis..."):
                t_start = time.perf_counter()

                # Kotak 9 s.d. 13: Retrieval
                t_ret_0 = time.perf_counter()
                retrieved_chunks = retriever.search(user_query, top_k=5)
                retrieval_ms = (time.perf_counter() - t_ret_0) * 1000.0

                if not retrieved_chunks:
                    st.error("⚠️ Tidak ditemukan pasal yang relevan dalam KUHP Baru.")
                    return

                # Kotak 14 s.d. 17: Reasoning & Guardrail
                t_gen_0 = time.perf_counter()
                legal_output = llm_engine.generate_legal_answer(user_query, retrieved_chunks)
                gen_s = time.perf_counter() - t_gen_0

                total_time_s = time.perf_counter() - t_start

                # Render AI Response
                st.markdown(legal_output.full_markdown)

                # Grounding Status Badge
                if legal_output.grounding_status:
                    st.markdown(
                        f"""
                        <div class="badge-grounded">
                            🛡️ Grounding Verified: 100% Sitasi Akurat ({len(legal_output.sitasi_valid)} Pasal Tervalidasi)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="badge-warning">
                            ⚠️ Peringatan Grounding: Terdapat rujukan yang tidak ada dalam konteks
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Latency Monitor
                st.caption(
                    f"⏱️ Waktu Eksekusi: Retrieval: {retrieval_ms:.1f} ms | LLM Reasoning: {gen_s:.2f} s | Total End-to-End: {total_time_s:.2f} s"
                )

                # 3. Transparent Legal Reference Expander (Explainability & Verifiability)
                render_reference_expander(retrieved_chunks)

                # Store Assistant Message in Session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": legal_output.full_markdown,
                    "retrieved_chunks": retrieved_chunks,
                })


def render_reference_expander(chunks: List[RetrievedLegalChunk]):
    """Render structured Top-K retrieved chunks with regulatory hierarchy and explanation."""
    with st.expander("📚 Referensi Pasal Asli (Klik untuk membaca)", expanded=False):
        st.markdown(f"**Ditemukan {len(chunks)} pasal relevan berdasarkan fusi hibrida (BM25 + FAISS):**")

        for c in chunks:
            buku_str = f"Buku: {c.buku}" if c.buku else ""
            bab_str = f" | Bab: {c.bab}" if c.bab else ""
            breadcrumb = f"{buku_str}{bab_str}"

            st.markdown(
                f"""
                <div class="chunk-box">
                    <div class="chunk-header">
                        Peringkat #{c.rank} — {c.pasal} <span style="font-size: 0.8rem; font-weight: normal; color: #64748b;">(Skor RRF: {c.score:.5f})</span>
                    </div>
                    <div class="chunk-breadcrumb">{breadcrumb}</div>
                    <div class="chunk-text"><b>Bunyi Norma Asli:</b><br>{c.text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Show authentic legislative explanation if substantive
            if c.penjelasan and c.penjelasan.strip() and c.penjelasan.strip().lower() != "cukup jelas":
                st.markdown(
                    f"""
                    <div class="chunk-penjelasan">
                        <b>📖 Penjelasan Autentik Pembentuk UU ({c.pasal}):</b><br>
                        {c.penjelasan}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()

