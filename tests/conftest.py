"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path

# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / "data"
TEST_FIXTURES_DIR = TEST_DATA_DIR / "fixtures"

# Ensure test directories exist
TEST_FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def sample_corpus() -> dict[str, str]:
    """Provide a small sample corpus for testing."""
    return {
        "Pasal 1": "Setiap orang bebas untuk memiliki keyakinan.",
        "Pasal 2": "Hak asasi manusia tidak dapat ditindak.",
        "Pasal 3": "Setiap orang berhak atas pengakuan atas hak.",
    }


@pytest.fixture
def sample_query() -> str:
    """Provide a sample query for testing."""
    return "Apakah seseorang bebas memiliki keyakinan?"
