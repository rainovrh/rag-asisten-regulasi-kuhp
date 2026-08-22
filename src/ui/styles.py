"""Custom CSS styles for Streamlit UI."""


def get_custom_styles() -> str:
    """Get custom CSS styles for the application.
    
    Returns:
        CSS string.
    """
    return """
    <style>
    :root {
        --navy-900: #0f1b2e;
        --navy-800: #1a2744;
        --navy-700: #243b5e;
        --navy-500: #345f8a;
        --gold-500: #c9a84c;
        --gold-400: #d4b85c;
        --gold-300: #e0c878;
        --ivory-50: #fdfcf7;
        --ivory-100: #f8f4e8;
        --ivory-200: #f1ebd6;
        --text-primary: #1a1a1a;
        --text-secondary: #4a4a4a;
        --text-muted: #6e6e6e;
        --border-subtle: #e8e3d5;
        --shadow-sm: 0 1px 3px rgba(15, 27, 46, 0.08);
        --shadow-md: 0 4px 12px rgba(15, 27, 46, 0.10);
        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
    }

    /* Base */
    .stApp {
        background-color: var(--ivory-50);
        color: var(--text-primary);
        font-family: "Source Serif 4", "Georgia", "Times New Roman", serif;
    }

    /* Header */
    h1, h2, h3, h4 {
        color: var(--navy-900);
        font-family: "Source Serif 4", "Georgia", "Times New Roman", serif;
        letter-spacing: -0.01em;
    }
    .stCaption {
        color: var(--text-muted);
    }

    /* Chat bubbles */
    .stChatMessage {
        padding: 1rem 1.15rem;
        border-radius: var(--radius-md);
        margin-bottom: 1rem;
        border: 1px solid var(--border-subtle);
        box-shadow: var(--shadow-sm);
    }
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, var(--navy-800), var(--navy-700));
        color: #ffffff;
        border-left: 4px solid var(--gold-500);
    }
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: var(--ivory-100);
        color: var(--text-primary);
        border-left: 4px solid var(--navy-500);
    }

    /* Chat input */
    .stChatInput > div {
        background-color: var(--ivory-50) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, var(--gold-500), var(--gold-400));
        color: var(--navy-900);
        border: none;
        border-radius: var(--radius-sm);
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: var(--shadow-sm);
        padding: 0.6rem 1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, var(--gold-400), var(--gold-300));
        box-shadow: var(--shadow-md);
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 700;
        color: var(--navy-800);
        background-color: var(--ivory-100);
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-subtle);
    }
    .streamlit-expanderContent {
        background-color: var(--ivory-50);
        border: 1px solid var(--border-subtle);
        border-top: none;
        border-radius: 0 0 var(--radius-sm) var(--radius-sm);
    }

    /* Info / warning / error boxes */
    .stInfo, .stWarning, .stSuccess, .stError {
        border-radius: var(--radius-sm);
        border-left: 4px solid var(--gold-500);
        background-color: var(--ivory-100);
        color: var(--text-primary);
        box-shadow: var(--shadow-sm);
    }
    .stWarning {
        border-left-color: #b45309;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--gold-500);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--navy-900), var(--navy-800));
        color: #ffffff;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: var(--gold-300);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span {
        color: #e6e1d5;
    }
    [data-testid="stSidebar"] .stDivider {
        border-color: rgba(201, 168, 76, 0.25);
    }
    [data-testid="stSidebar"] .stCaption {
        color: #c8c0aa;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--ivory-100);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--navy-500);
        border-radius: 8px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--navy-700);
    }
    </style>
    """
