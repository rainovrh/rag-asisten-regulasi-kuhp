"""Path management for the application."""

from pathlib import Path

# Base directory (File Hukum/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEXES_DIR = DATA_DIR / "indexes"
DATASETS_DIR = DATA_DIR / "datasets"

# Source directories
SRC_DIR = BASE_DIR / "src"
CONFIG_DIR = SRC_DIR / "config"
PREPROCESSING_DIR = SRC_DIR / "preprocessing"
RETRIEVAL_DIR = SRC_DIR / "retrieval"
GENERATION_DIR = SRC_DIR / "generation"
EVALUATION_DIR = SRC_DIR / "evaluation"
UI_DIR = SRC_DIR / "ui"
UTILS_DIR = SRC_DIR / "utils"

# Other directories
LOGS_DIR = BASE_DIR / "logs"
TESTS_DIR = BASE_DIR / "tests"
SCRIPTS_DIR = BASE_DIR / "scripts"
DOCS_DIR = BASE_DIR / "docs"

# Ensure directories exist
for directory in [
    RAW_DATA_DIR, PROCESSED_DATA_DIR, INDEXES_DIR, DATASETS_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
