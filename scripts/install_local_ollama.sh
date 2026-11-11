#!/bin/bash
set -euo pipefail

# Local Ollama Setup for RAG-Asisten-Regulasi
# Pastikan Ollama diinstal (https://ollama.ai)
# Install model yang diperlukan: qwen2.5:7b-instruct

echo "Memeriksa Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Ollama tidak ditemukan. Unduh dari https://ollama.ai"
    echo "Setelah diinstal, jalankan skrip ini lagi."
    exit 1
fi

echo "Mengunduh qwen2.5:7b-instruct (ca. 5GB)..."
ollama pull qwen2.5:7b-instruct

echo "Setup selesai."
echo "Sekarang Anda dapat mengubah LLM_PROVIDER dan LLM_MODEL di .env:

LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b-instruct
"
