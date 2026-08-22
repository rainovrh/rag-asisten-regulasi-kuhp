#!/usr/bin/env python3
"""Verify migration completeness and system readiness."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def check_directory_structure() -> bool:
    """Check that all required directories exist."""
    required_dirs = [
        "src/config",
        "src/preprocessing",
        "src/retrieval",
        "src/generation",
        "src/evaluation",
        "src/ui",
        "src/utils",
        "tests",
        "data/raw",
        "data/processed",
        "data/indexes",
        "data/datasets",
        "scripts",
        "docs",
        "logs",
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        full_path = BASE_DIR / dir_path
        if not full_path.exists():
            print(f"[MISSING] {dir_path}")
            all_ok = False
        else:
            print(f"[OK] {dir_path}")
    
    return all_ok


def check_core_modules() -> bool:
    """Check that core Python modules exist."""
    required_files = [
        "src/__init__.py",
        "src/config/__init__.py",
        "src/config/settings.py",
        "src/config/paths.py",
        "src/config/constants.py",
        "src/preprocessing/__init__.py",
        "src/preprocessing/cleaner.py",
        "src/preprocessing/normalizer.py",
        "src/preprocessing/chunker.py",
        "src/retrieval/__init__.py",
        "src/retrieval/bm25_retriever.py",
        "src/retrieval/dense_retriever.py",
        "src/retrieval/hybrid_retriever.py",
        "src/retrieval/reranker.py",
        "src/generation/__init__.py",
        "src/generation/llm.py",
        "src/generation/prompts.py",
        "src/evaluation/__init__.py",
        "src/evaluation/metrics.py",
        "src/evaluation/golden_dataset.py",
        "src/evaluation/runner.py",
        "src/ui/__init__.py",
        "src/ui/app.py",
        "src/ui/components.py",
        "src/ui/styles.py",
        "src/utils/__init__.py",
        "src/utils/logger.py",
        "src/utils/validators.py",
        "src/utils/helpers.py",
        "src/utils/exceptions.py",
        "tests/__init__.py",
        "tests/conftest.py",
        "tests/test_preprocessing.py",
        "tests/test_retrieval.py",
        "tests/test_generation.py",
        "tests/test_evaluation.py",
    ]
    
    all_ok = True
    for file_path in required_files:
        full_path = BASE_DIR / file_path
        if not full_path.exists():
            print(f"[MISSING] Missing file: {file_path}")
            all_ok = False
        else:
            print(f"[OK] {file_path}")
    
    return all_ok


def check_config_files() -> bool:
    """Check configuration files."""
    required_files = [
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        ".gitignore",
        ".env.example",
        "README.md",
        "CONTRIBUTING.md",
        "STYLE_GUIDE.md",
    ]
    
    all_ok = True
    for file_path in required_files:
        full_path = BASE_DIR / file_path
        if not full_path.exists():
            print(f"[MISSING] Missing file: {file_path}")
            all_ok = False
        else:
            print(f"[OK] {file_path}")
    
    return all_ok


def check_scripts() -> bool:
    """Check runner scripts."""
    required_scripts = [
        "scripts/run_app.py",
        "scripts/run_preprocessing.py",
        "scripts/run_indexing.py",
        "scripts/run_evaluation.py",
    ]
    
    all_ok = True
    for script in required_scripts:
        full_path = BASE_DIR / script
        if not full_path.exists():
            print(f"[MISSING] Missing script: {script}")
            all_ok = False
        else:
            print(f"[OK] {script}")
    
    return all_ok


def check_data_files() -> bool:
    """Check data files (warnings only, not required for verification)."""
    data_files = [
        "data/raw/KUHP BARU UU Nomor 1 Tahun 2023.pdf",
        "data/processed/kuhp_bersih.json",
        "data/indexes/faiss_index_kuhp/index.faiss",
        "data/datasets/golden_dataset_rag_hukum_indonesia_rev3.csv",
    ]
    
    print("\n--- Data Files (Warnings Only) ---")
    for file_path in data_files:
        full_path = BASE_DIR / file_path
        if not full_path.exists():
            print(f"[WARN]  Missing data file: {file_path}")
        else:
            print(f"[OK] {file_path}")
    
    return True


def main() -> None:
    """Run all verification checks."""
    print("=" * 60)
    print("MIGRATION VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Core Modules", check_core_modules),
        ("Configuration Files", check_config_files),
        ("Runner Scripts", check_scripts),
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        if not check_func():
            all_passed = False
    
    check_data_files()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[OK] ALL CHECKS PASSED - Migration complete!")
    else:
        print("[MISSING] SOME CHECKS FAILED - Please review missing items")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
