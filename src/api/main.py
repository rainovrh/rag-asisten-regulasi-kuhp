import json
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config.settings import settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.generation.llm import LLMEngine
from src.generation.prompts import get_legal_qa_prompt


# Global instances
retriever = None
llm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm
    print("Loading KUHP Knowledge Base and AI Models...")
    try:
        corpus_path = settings.processed_corpus_path
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)

        bm25 = BM25Retriever(corpus_data=corpus_data)
        dense = DenseRetriever(index_path=settings.faiss_index_path)
        retriever = HybridRetriever(bm25, dense)
        llm = LLMEngine()
        print("System loaded successfully.")
    except Exception as e:
        print(f"Error loading system: {e}")
    yield


app = FastAPI(title="RAG KUHP API", lifespan=lifespan)

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str


class Citation(BaseModel):
    pasal: str
    skor: float
    teks: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    is_refusal: bool


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if retriever is None or llm is None:
        raise HTTPException(status_code=503, detail="System is still initializing.")

    # Retrieve context
    retrieved_data = retriever.search(request.query, top_k=5, apply_crag=True)

    if not retrieved_data:
        return ChatResponse(
            answer="⚠️ **Informasi Tidak Ditemukan dalam KUHP Baru.**\n\nSistem membuang konteks karena tidak ditemukan pasal yang memiliki tingkat relevansi memadai.",
            citations=[],
            is_refusal=True
        )

    context_str = "\n\n---\n\n".join(text for doc_id, _, text in retrieved_data)
    prompt = get_legal_qa_prompt()
    
    # Generate answer
    try:
        ai_response = llm.generate_with_template(prompt, context_str, request.query)
        is_refusal = llm.is_refusal(ai_response)
        
        if is_refusal:
            ai_response = "⚠️ **Informasi Tidak Ditemukan dalam KUHP Baru.**\n\nSistem tidak dapat menemukan pasal yang relevan untuk menjawab pertanyaan ini berdasarkan konteks yang tersedia."
            citations = []
        else:
            citations = [
                Citation(pasal=doc_id, skor=score, teks=text)
                for doc_id, score, text in retrieved_data
            ]
            
        return ChatResponse(
            answer=ai_response,
            citations=citations,
            is_refusal=is_refusal
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    return {"status": "ok"}
