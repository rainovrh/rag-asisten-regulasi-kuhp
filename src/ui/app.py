"""Main Streamlit application."""

import sys
from pathlib import Path

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config.constants import CHAT_INPUT_PLACEHOLDER
from src.ui.components import (
    init_page_config,
    render_header,
    render_chat_message,
    render_processing_status,
    render_no_context_message,
    render_sidebar,
)
from src.ui.styles import get_custom_styles


def init_session_state() -> None:
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar() -> None:
    """Render sidebar with app information."""
    with st.sidebar:
        st.header("📋 Tentang Aplikasi")
        st.markdown("""
        **Asisten Hukum AI (RAG KUHP)** adalah sistem berbasis AI yang 
        membantu mencari dan memahami pasal-pasal dalam KUHP Baru 
        (UU No. 1 Tahun 2023).
        
        ### Fitur Utama:
        - Pencarian hibrida (BM25 + Semantic)
        - Reciprocal Rank Fusion (RRF)
        - CRAG Filtering untuk keandalan
        - Sitasi pasal terverifikasi
        
        ### Batasan:
        - Sistem hanya menjawab berdasarkan KUHP Baru
        - Hasil AI memerlukan verifikasi ahli hukum
        - Tidak memberikan nasihat hukum resmi
        """)
        
        st.divider()
        st.markdown("**⚠️ Disclaimer**")
        st.caption(
            "Sistem ini adalah prototype akademis. "
            "Hasil analisis tidak menggantikan konsultasi dengan ahli hukum."
        )
        
        st.divider()
        st.markdown("© 2026 - Rainova Rahaniawan")
        st.markdown("Institut Teknologi Nasional (ITENAS)")


def main() -> None:
    """Main application entry point."""
    init_page_config()
    init_session_state()
    render_header()
    render_sidebar()
    
    # Apply custom styles
    st.markdown(get_custom_styles(), unsafe_allow_html=True)
    
    # Load system components (cached)
    with st.spinner("Memuat Basis Pengetahuan KUHP dan Model AI..."):
        from src.config.settings import settings
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.dense_retriever import DenseRetriever
        from src.generation.llm import LLMEngine
        from src.generation.prompts import get_legal_qa_prompt
        
        @st.cache_resource
        def load_system():
            """Load and cache system components."""
            # Load corpus
            import json
            from pathlib import Path
            
            corpus_path = settings.processed_corpus_path
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus_data = json.load(f)
            
            # Initialize retrievers
            bm25 = BM25Retriever(corpus_data=corpus_data)
            dense = DenseRetriever(index_path=settings.faiss_index_path)
            retriever = HybridRetriever(bm25, dense)
            
            # Initialize LLM
            llm = LLMEngine()
            
            return retriever, llm
        
        retriever, llm = load_system()
    
    # Render chat history
    for msg in st.session_state.messages:
        render_chat_message(
            role=msg["role"],
            content=msg["content"],
            context_used=msg.get("context_used"),
        )
    
    # Chat input
    if user_query := st.chat_input(CHAT_INPUT_PLACEHOLDER):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        render_chat_message(role="user", content=user_query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Mencari pasal KUHP dan Menganalisis..."):
                retrieved_data = retriever.search(user_query, top_k=5, apply_crag=True)
                
                if not retrieved_data:
                    render_no_context_message()
                    ai_response = (
                        "⚠️ **Informasi Tidak Ditemukan dalam KUHP Baru.**\n\n"
                        "Sistem (Filter CRAG) membuang konteks karena tidak ditemukan "
                        "pasal yang memiliki tingkat relevansi memadai."
                    )
                else:
                    context_str = "\n\n".join(
                        f"[{doc_id}]: {text}" for doc_id, _, text in retrieved_data
                    )
                    prompt = get_legal_qa_prompt()
                    ai_response = llm.generate_with_template(prompt, context_str, user_query)
                    
                    if llm.is_refusal(ai_response):
                        ai_response = (
                            "⚠️ **Informasi Tidak Ditemukan dalam KUHP Baru.**\n\n"
                            "Sistem tidak dapat menemukan pasal yang relevan untuk "
                            "menjawab pertanyaan ini berdasarkan konteks yang tersedia."
                        )
                
                st.markdown(ai_response)
                
                # Show context
                if retrieved_data:
                    with st.expander("🔍 Lihat Referensi Pasal Terverifikasi (Traceability)"):
                        for doc_id, score, text in retrieved_data:
                            st.info(
                                f"**{doc_id}** (Skor Keandalan RRF: {score:.4f})\n\n{text}"
                            )
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_response,
            "context_used": [
                {"pasal": doc_id, "skor": score, "teks": text}
                for doc_id, score, text in (retrieved_data or [])
            ],
        })


if __name__ == "__main__":
    main()
