"""Streamlit UI components."""

import streamlit as st

from src.config.constants import (
    DEFAULT_PAGE_TITLE,
    CHAT_INPUT_PLACEHOLDER,
    PROCESSING_MESSAGE,
)


def init_page_config() -> None:
    """Initialize Streamlit page configuration."""
    st.set_page_config(
        page_title=DEFAULT_PAGE_TITLE,
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header() -> None:
    """Render application header."""
    st.title("⚖️ Asisten Hukum Berbasis AI (KUHP Baru)")
    st.caption(
        "Human-Centered AI: Jawaban dihasilkan berbasis rujukan pasal resmi "
        "KUHP (UU No. 1 Tahun 2023)."
    )


def render_chat_message(role: str, content: str, context_used: list | None = None) -> None:
    """Render a single chat message.
    
    Args:
        role: Message role ('user' or 'assistant').
        content: Message content.
        context_used: Optional list of context chunks used.
    """
    with st.chat_message(role):
        st.markdown(content)
        
        if context_used:
            with st.expander("🔍 Lihat Referensi Pasal Terverifikasi (Traceability)"):
                for ctx in context_used:
                    st.info(
                        f"**{ctx['pasal']}** (Skor Keandalan RRF: {ctx['skor']:.4f})\n\n"
                        f"{ctx['teks']}"
                    )


def render_processing_status(message: str = PROCESSING_MESSAGE) -> None:
    """Render processing status spinner.
    
    Args:
        message: Status message to display.
    """
    with st.spinner(message):
        yield


def render_no_context_message() -> None:
    """Render message when no context is found."""
    st.warning(
        "⚠️ **Informasi Tidak Ditemukan dalam KUHP Baru.**\n\n"
        "Sistem (Filter CRAG) membuang konteks karena tidak ditemukan pasal "
        "yang memiliki tingkat relevansi memadai. Untuk menjaga keandalan "
        "hukum dan mencegah halusinasi AI, analisis tidak diberikan."
    )


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
