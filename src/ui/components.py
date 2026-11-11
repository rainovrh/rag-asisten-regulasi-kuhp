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
    st.markdown(
        """
        <div class="app-heading">
            <p class="app-heading__eyebrow">Asisten Hukum AI</p>
            <h1 class="app-heading__title">⚖️ Asisten Hukum Berbasis AI (KUHP Baru)</h1>
            <p class="app-heading__subtitle">
                Jawaban dihasilkan berbasis rujukan pasal resmi KUHP (UU No. 1 Tahun 2023).
                Sistem menggunakan pencarian hibrida (BM25 + Semantic), RRF, dan CRAG
                untuk keandalan hukum.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_citations(context_used: list) -> None:
    """Render citation cards below the AI response.

    Args:
        context_used: List of dicts with keys 'pasal', 'skor', 'teks'.
    """
    import re

    if not context_used:
        return

    st.markdown(
        """
        <div class=\"citation-section\">
            <p class=\"citation-heading\">📎 Referensi Pasal Terverifikasi</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for i, ctx in enumerate(context_used, start=1):
        raw_id = ctx["pasal"]
        skor = ctx["skor"]
        teks = ctx["teks"]

        # Clean label: strip "_chunk_X" suffix and normalize
        clean_id = re.sub(r"[_\-]?chunk[_\-]?\d+", "", raw_id, flags=re.IGNORECASE)
        clean_id = clean_id.replace("_", " ").strip()
        # Capitalize "Pasal" and number nicely
        parts = clean_id.split()
        pasal_label = " ".join(p.capitalize() for p in parts) if parts else raw_id

        # Determine badge type
        if skor >= 0.05:
            badge_class = "citation-badge--high"
            badge_label = "Relevansi Tinggi"
        elif skor >= 0.02:
            badge_class = "citation-badge--medium"
            badge_label = "Relevansi Sedang"
        else:
            badge_class = "citation-badge"
            badge_label = "Relevansi Rendah"

        # Build Google search URL for this pasal
        import urllib.parse
        search_query = urllib.parse.quote(f"{pasal_label} KUHP Baru UU Nomor 1 Tahun 2023")
        search_url = f"https://www.google.com/search?q={search_query}"

        # Escape HTML characters in teks to prevent injection / rendering bugs
        import html
        safe_teks = html.escape(teks)

        st.markdown(
            f"""
            <a href=\"{search_url}\" target=\"_blank\" rel=\"noopener noreferrer\" class=\"citation-card\">
              <div class=\"citation-row\">
                <span class=\"citation-title\">[{i}] {pasal_label}</span>
                <span class=\"{badge_class}\">{badge_label} · Skor {skor:.4f}</span>
              </div>
              <div class=\"citation-text\">{safe_teks}</div>
            </a>
            """,
            unsafe_allow_html=True,
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
        
        if context_used and role == "assistant":
            render_citations(context_used)


def render_processing_status(message: str = PROCESSING_MESSAGE) -> None:
    """Render processing status spinner.
    
    Args:
        message: Status message to display.
    """
    with st.spinner(message):
        yield


def render_no_context_message() -> None:
    """Render message when no context is found."""
    st.markdown(
        """
        <div class="stWarning">
            <strong>⚠️ Informasi Tidak Ditemukan dalam KUHP Baru.</strong>
            <p style="margin:0.5rem 0 0">
                Sistem (Filter CRAG) membuang konteks karena tidak ditemukan pasal yang
                memiliki tingkat relevansi memadai. Untuk menjaga keandalan hukum dan
                mencegah halusinasi AI, analisis tidak diberikan.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render sidebar with app information - clean minimal design."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-footer">
                <p>📋 <strong>Tentang Aplikasi</strong></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <ul class="sidebar-list">
                <li><strong>Asisten Hukum AI (RAG KUHP)</strong></li>
                <li>Sistem berbasis AI untuk pencarian dan pemahaman pasal KUHP Baru (UU No. 1/2023)</li>
                <li style="margin-top:1rem"><strong>Fitur Utama:</strong></li>
                <li>Pencarian hibrida (BM25 + Semantic)</li>
                <li>Reciprocal Rank Fusion (RRF)</li>
                <li>CRAG Filtering untuk keandalan</li>
                <li>Sitasi pasal terverifikasi</li>
                <li style="margin-top:1rem"><strong>Batasan:</strong></li>
                <li>Tidak memberikan nasihat hukum resmi</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr class=\"sidebar-divider\">", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="app-note">
                ⚠️ <strong>Disclaimer</strong><br>
                Sistem ini adalah prototype akademis. Hasil analisis tidak menggantikan
                konsultasi dengan ahli hukum.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <hr class="sidebar-divider">
            <div class="sidebar-footer">
                © 2026 Rainova Rahaniawan<br>
                Institut Teknologi Nasional (ITENAS)
            </div>
            """,
            unsafe_allow_html=True,
        )
