"""Minimal, modern styling for the Streamlit application."""


def get_custom_styles() -> str:
    """Return the application's global CSS."""
    return """
    <style>
    :root {
        --slate-950: #0f172a;
        --slate-900: #1e293b;
        --slate-800: #334155;
        --slate-700: #475569;
        --slate-600: #64748b;
        --slate-500: #94a3b8;
        --slate-300: #cbd5e1;
        --slate-200: #e2e8f0;
        --slate-100: #f1f5f9;
        --slate-50: #f8fafc;

        --blue-700: #1d4ed8;
        --blue-600: #2563eb;
        --blue-500: #3b82f6;
        --blue-100: #dbeafe;
        --blue-50: #eff6ff;

        --green-600: #15803d;
        --green-100: #dcfce7;
        --amber-600: #b45309;
        --amber-100: #fef3c7;
        --gray-600: #475569;
        --gray-100: #f1f5f9;

        --white: #ffffff;
        --text-primary: #0f172a;
        --text-secondary: #334155;
        --text-muted: #64748b;
        --border-subtle: #e2e8f0;
        --border-strong: #cbd5e1;

        --radius-xs: 4px;
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 10px;
        --radius-xl: 12px;
        --radius-pill: 999px;

        --shadow-xs: 0 1px 2px rgba(15, 23, 42, 0.04);
        --shadow-sm: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
        --shadow-md: 0 4px 12px rgba(15, 23, 42, 0.08), 0 2px 4px rgba(15, 23, 42, 0.04);
        --shadow-lg: 0 8px 24px rgba(15, 23, 42, 0.10), 0 4px 8px rgba(15, 23, 42, 0.05);

        --font-sans: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        --ease: cubic-bezier(0.2, 0, 0, 1);
    }

    * {
        box-sizing: border-box;
    }

    .stApp {
        background: var(--slate-50);
        color: var(--text-primary);
        font-family: var(--font-sans);
        font-size: 15px;
        line-height: 1.65;
        -webkit-font-smoothing: antialiased;
    }

    .stApp::before {
        content: "";
        position: fixed;
        z-index: -1;
        inset: 0;
        background:
            radial-gradient(circle at 90% 0%, rgba(59, 130, 246, 0.055), transparent 28rem),
            radial-gradient(circle at 0% 100%, rgba(15, 23, 42, 0.035), transparent 28rem);
        pointer-events: none;
    }

    .main .block-container {
        padding: 2rem 2.5rem 4rem 2.5rem !important;
        max-width: 920px !important;
    }

    h1, h2, h3, h4, h5, h6, p, li, span {
        font-family: var(--font-sans);
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--slate-950);
        font-weight: 650;
        letter-spacing: -0.02em;
        line-height: 1.3;
    }

    p, li {
        color: var(--text-secondary);
    }

    strong {
        color: var(--slate-950);
        font-weight: 650;
    }

    code {
        background: var(--slate-100);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-xs);
        color: var(--slate-800);
        font-size: 0.85em;
        padding: 0.15em 0.4em;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
        font-style: normal !important;
        line-height: 1.6 !important;
    }

    /* Header */
    .app-heading {
        margin-bottom: 1.75rem;
    }

    .app-heading__eyebrow {
        color: var(--blue-600);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin: 0 0 0.35rem;
        text-transform: uppercase;
    }

    .app-heading__title {
        color: var(--slate-950) !important;
        font-size: clamp(1.65rem, 4vw, 2.35rem) !important;
        font-weight: 700 !important;
        letter-spacing: -0.035em !important;
        line-height: 1.12 !important;
        margin: 0;
    }

    .app-heading__subtitle {
        color: var(--text-muted) !important;
        font-size: 0.96rem !important;
        line-height: 1.65 !important;
        margin: 0.55rem 0 0;
        max-width: 720px;
    }

    .app-section-title {
        color: var(--slate-950) !important;
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        margin: 1.5rem 0 0.6rem;
        text-transform: uppercase;
    }

    .app-note {
        background: var(--blue-50);
        border: 1px solid var(--blue-100);
        border-radius: var(--radius-md);
        color: var(--slate-700);
        font-size: 0.82rem;
        line-height: 1.6;
        margin: 0.75rem 0 0;
        padding: 0.7rem 0.9rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--slate-950) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1.25rem !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: var(--white) !important;
        font-size: 1rem !important;
        font-weight: 650 !important;
        letter-spacing: -0.01em !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li {
        color: #cbd5e1 !important;
        font-size: 0.82rem !important;
        line-height: 1.65 !important;
    }

    [data-testid="stSidebar"] strong {
        color: var(--white) !important;
    }

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 1.25rem 0 !important;
    }

    .sidebar-list {
        margin: 0.35rem 0 0 !important;
        padding-left: 1.1rem !important;
    }

    .sidebar-list li {
        margin-bottom: 0.38rem !important;
        padding-left: 0.1rem !important;
    }

    .sidebar-divider {
        border-color: rgba(255, 255, 255, 0.08) !important;
    }

    .sidebar-footer {
        color: #94a3b8 !important;
        font-size: 0.76rem !important;
        line-height: 1.55 !important;
        margin-top: 0.7rem !important;
    }

    /* Chat */
    .stChatMessage {
        border: 1px solid transparent !important;
        border-radius: var(--radius-lg) !important;
        margin-bottom: 1rem !important;
        padding: 1rem 1.25rem !important;
    }

    .stChatMessage[data-testid="stChatMessageUser"] {
        background: var(--slate-900) !important;
        border-left: 3px solid var(--blue-500) !important;
        border-top-left-radius: var(--radius-sm) !important;
        color: var(--white) !important;
    }

    .stChatMessage[data-testid="stChatMessageUser"] p,
    .stChatMessage[data-testid="stChatMessageUser"] span {
        color: var(--white) !important;
    }

    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background: var(--white) !important;
        border-color: var(--border-subtle) !important;
        border-left: 3px solid var(--slate-300) !important;
        border-top-right-radius: var(--radius-sm) !important;
        box-shadow: var(--shadow-xs) !important;
    }

    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background: var(--blue-500) !important;
        border-radius: var(--radius-pill) !important;
    }

    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background: var(--slate-700) !important;
        border-radius: var(--radius-pill) !important;
    }

    .stChatInput {
        padding-top: 1.25rem !important;
    }

    .stChatInput > div,
    [data-testid="stChatInput"] > div {
        background: var(--white) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-xl) !important;
        box-shadow: var(--shadow-xs) !important;
        padding: 0.25rem 0.5rem !important;
        transition: border-color 0.15s var(--ease), box-shadow 0.15s var(--ease) !important;
    }

    .stChatInput > div:focus-within,
    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--blue-500) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.10), var(--shadow-sm) !important;
    }

    .stChatInput textarea {
        color: var(--text-primary) !important;
        font-family: var(--font-sans) !important;
        font-size: 0.95rem !important;
    }

    .stChatInput button {
        background: var(--blue-500) !important;
        border: none !important;
        border-radius: var(--radius-pill) !important;
        transition: background 0.15s var(--ease), transform 0.15s var(--ease) !important;
    }

    .stChatInput button:hover {
        background: var(--blue-600) !important;
        transform: scale(1.04) !important;
    }

    .stChatInput button:active {
        transform: scale(0.96) !important;
    }

    /* Native Streamlit controls */
    .stButton > button {
        background: var(--blue-500) !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-xs) !important;
        color: var(--white) !important;
        font-family: var(--font-sans) !important;
        font-size: 0.88rem !important;
        font-weight: 550 !important;
        padding: 0.6rem 1.4rem !important;
        transition: background 0.15s var(--ease), box-shadow 0.15s var(--ease), transform 0.15s var(--ease) !important;
    }

    .stButton > button:hover {
        background: var(--blue-600) !important;
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button:active {
        background: var(--blue-700) !important;
        box-shadow: var(--shadow-xs) !important;
        transform: translateY(0) !important;
    }

    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stTextArea > div > div,
    .stNumberInput > div > div {
        background: var(--white) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-xs) !important;
        font-family: var(--font-sans) !important;
        transition: border-color 0.15s var(--ease), box-shadow 0.15s var(--ease) !important;
    }

    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div:focus-within,
    .stTextArea > div > div:focus-within {
        border-color: var(--blue-500) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.10) !important;
    }

    [data-baseweb="popover"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-lg) !important;
        overflow: hidden;
    }

    [data-baseweb="option"] {
        border-radius: var(--radius-xs) !important;
    }

    [data-baseweb="option"]:hover {
        background: var(--slate-100) !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: var(--white) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        box-shadow: var(--shadow-xs) !important;
        margin-top: 0.75rem !important;
        overflow: hidden;
        transition: border-color 0.15s var(--ease), box-shadow 0.15s var(--ease) !important;
    }

    [data-testid="stExpander"]:hover {
        border-color: var(--blue-100) !important;
        box-shadow: var(--shadow-md) !important;
    }

    [data-testid="stExpander"] summary,
    .streamlit-expanderHeader {
        color: var(--slate-950) !important;
        cursor: pointer;
        font-size: 0.88rem !important;
        font-weight: 650 !important;
        padding: 0.85rem 1.1rem !important;
    }

    [data-testid="stExpander"] summary:hover {
        background: var(--blue-50) !important;
        color: var(--blue-600) !important;
    }

    .streamlit-expanderContent,
    [data-testid="stExpander"] > div:last-child {
        background: var(--white) !important;
        border-top: 1px solid var(--border-subtle) !important;
        border-radius: 0 0 var(--radius-lg) var(--radius-lg) !important;
        padding: 0.9rem 1.1rem 1.1rem !important;
    }

    /* Alerts and status */
    .stAlert, [data-testid="stAlert"],
    .stInfo, .stWarning, .stSuccess, .stError {
        background: var(--white) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-xs) !important;
        font-family: var(--font-sans) !important;
        font-size: 0.85rem !important;
        line-height: 1.65 !important;
        padding: 0.9rem 1.1rem !important;
    }

    .stInfo {
        background: var(--blue-50) !important;
        border-color: var(--blue-100) !important;
        border-left: 3px solid var(--blue-500) !important;
        color: var(--slate-700) !important;
    }

    .stWarning {
        background: var(--amber-100) !important;
        border-color: #fde68a !important;
        border-left: 3px solid #d97706 !important;
        color: #78350f !important;
    }

    .stSuccess {
        background: var(--green-100) !important;
        border-color: #bbf7d0 !important;
        border-left: 3px solid #16a34a !important;
        color: #14532d !important;
    }

    .stError {
        background: #fef2f2 !important;
        border-color: #fecaca !important;
        border-left: 3px solid #ef4444 !important;
        color: #7f1d1d !important;
    }

    .stSpinner > div {
        border-left-color: var(--blue-500) !important;
        border-right-color: var(--blue-500) !important;
        border-top-color: var(--blue-500) !important;
    }

    [data-testid="stStatusWidget"] {
        background: var(--white) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-xs) !important;
    }

    /* Metrics, dividers, tabs, progress */
    [data-testid="stMetric"] {
        background: var(--white) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-lg) !important;
        box-shadow: var(--shadow-xs) !important;
        padding: 1rem 1.25rem !important;
        transition: border-color 0.15s var(--ease), box-shadow 0.15s var(--ease) !important;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--blue-100) !important;
        box-shadow: var(--shadow-md) !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
        font-weight: 550 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--slate-950) !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    hr, .stDivider {
        background: var(--border-subtle) !important;
        border: none !important;
        border-radius: var(--radius-pill) !important;
        height: 1px !important;
        margin: 1.5rem 0 !important;
    }

    [data-baseweb="tab-list"] {
        background: var(--slate-100) !important;
        border-radius: var(--radius-pill) !important;
        gap: 2px !important;
        padding: 3px !important;
    }

    [data-baseweb="tab"] {
        border-radius: var(--radius-pill) !important;
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
        font-weight: 550 !important;
        padding: 0.35rem 1rem !important;
    }

    [aria-selected="true"][data-baseweb="tab"] {
        background: var(--white) !important;
        box-shadow: var(--shadow-sm) !important;
        color: var(--slate-950) !important;
    }

    .stProgress > div > div > div {
        background: var(--blue-500) !important;
        border-radius: var(--radius-pill) !important;
    }

    .stProgress > div > div {
        background: var(--slate-100) !important;
        border-radius: var(--radius-pill) !important;
        height: 5px !important;
    }

    /* Citation cards */
    .citation-section {
        margin-top: 1.4rem !important;
    }

    .citation-heading {
        color: var(--text-muted) !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        margin: 0 0 0.7rem !important;
        text-transform: uppercase;
    }

    .citation-card {
        background: var(--white) !important;
        border: 1px solid var(--border-subtle) !important;
        border-left: 3px solid var(--slate-300) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-xs) !important;
        color: inherit !important;
        display: block !important;
        margin: 0.6rem 0 !important;
        padding: 0.9rem 1.1rem !important;
        text-decoration: none !important;
        transition: border-color 0.15s var(--ease), box-shadow 0.15s var(--ease) !important;
    }

    .citation-card:hover {
        border-color: var(--blue-100) !important;
        box-shadow: var(--shadow-md) !important;
    }

    .citation-row {
        align-items: center;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: space-between;
        margin-bottom: 0.55rem;
    }

    .citation-title {
        color: var(--slate-900) !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
    }

    .citation-badge {
        background: var(--gray-100) !important;
        border-radius: var(--radius-pill) !important;
        color: var(--gray-600) !important;
        font-size: 0.68rem !important;
        font-weight: 650 !important;
        padding: 0.2rem 0.65rem !important;
        white-space: nowrap;
    }

    .citation-badge--high {
        background: var(--green-100) !important;
        color: var(--green-600) !important;
    }

    .citation-badge--medium {
        background: var(--amber-100) !important;
        color: var(--amber-600) !important;
    }

    .citation-text {
        color: var(--slate-700) !important;
        font-size: 0.82rem !important;
        line-height: 1.7 !important;
        white-space: pre-wrap !important;
        word-break: break-word;
    }

    .citation-link {
        color: var(--blue-600) !important;
        font-size: 0.78rem !important;
        font-weight: 650 !important;
        margin-top: 0.55rem !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        height: 6px;
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
        border-radius: var(--radius-pill);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--slate-300);
        border-radius: var(--radius-pill);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--slate-500);
    }

    :focus-visible {
        border-radius: var(--radius-sm) !important;
        outline: 2px solid var(--blue-400) !important;
        outline-offset: 2px !important;
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding: 1.25rem 1rem 3rem 1rem !important;
        }

        .app-heading {
            margin-bottom: 1.25rem;
        }

        .app-heading__subtitle {
            font-size: 0.9rem !important;
        }

        .stChatMessage {
            padding: 0.85rem 1rem !important;
        }

        .stChatInput > div,
        [data-testid="stChatInput"] > div {
            border-radius: var(--radius-lg) !important;
        }

        .citation-row {
            align-items: flex-start;
            flex-direction: column;
        }

        .citation-badge {
            white-space: normal;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """
