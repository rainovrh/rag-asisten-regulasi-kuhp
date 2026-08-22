# TECHNICAL ASSESSMENT: Backend Architecture Decision

## Executive Summary

**RECOMMENDATION: Dedicated backend is NOT required for the current thesis scope.**

The project should remain as an **enhanced monolithic Streamlit application** with lightweight session-based authentication and file-based persistence. Building a dedicated backend (FastAPI/Flask + PostgreSQL) would introduce unnecessary complexity that distracts from the core research contribution: evaluating Hybrid Retrieval and Semantic Chunking for Indonesian legal RAG systems.

---

## Current State Analysis

### Existing Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRENT ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────┤
│  Frontend:     Streamlit (`src/ui/app.py` - modular architecture)  │
│  Vector DB:    FAISS (file-based, local)                        │
│  LLM:          Ollama + Llama-3 8B (local inference)            │
│  Corpus:       kuhp_bersih.json (static file)                   │
│  Evaluation:   CSV-based batch scripts (fase4, fase5)           │
│  Auth:         None (open local access)                         │
│  Database:     None (file-based storage only)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Characteristics
1. **Single-user, local execution** on Asus Vivobook Pro 14 OLED
2. **File-based persistence** (JSON, CSV, FAISS index files)
3. **Local LLM inference** via Ollama (no external API calls)
4. **Research prototype** scope (not production deployment)
5. **Batch processing** via standalone Python scripts

---

## Technical Justification: Why No Dedicated Backend

### 1. Research Scope Alignment

**Thesis Contribution**: The core intellectual contribution is the **algorithmic evaluation** of:
- Hybrid Retrieval (BM25 + Dense + RRF)
- Semantic Chunking vs Fixed-size Chunking
- Faithfulness and Answer Relevance metrics (RAGAS)

**Backend Contribution**: A dedicated backend contributes **zero** to these research questions. It is infrastructure, not research.

**Verdict**: Allocate development time to algorithm refinement and evaluation, not web architecture.

### 2. Current Architecture Sufficiency

The existing monolithic Streamlit architecture already supports:
- ✅ Document upload and preprocessing
- ✅ Vector database management (FAISS)
- ✅ Hybrid retrieval with RRF
- ✅ LLM inference (Ollama)
- ✅ Batch evaluation with Golden Dataset
- ✅ RAGAS metrics calculation
- ✅ Interactive chat interface

**Missing**: Only RBAC and audit logging, which can be added to Streamlit without a backend.

### 3. Data Characteristics Analysis

| Data Type | Current Storage | Backend Need? | Justification |
|-----------|-----------------|---------------|---------------|
| Legal Corpus | JSON file | ❌ No | Static, read-only, loaded once at startup |
| Vector Embeddings | FAISS index | ❌ No | File-based, no concurrent writes |
| Golden Dataset | CSV | ❌ No | Batch processing, not real-time |
| Evaluation Results | CSV | ❌ No | Append-only, no complex queries |
| User Data | None | ⚠️ Minimal | 3 roles, <10 users, simple permissions |

**Conclusion**: No data characteristics require relational database features (ACID transactions, complex joins, concurrent writes).

### 4. Performance & Scalability

| Aspect | Current | With Backend | Impact |
|--------|---------|--------------|--------|
| Inference Latency | ~5-10s | ~5-10s + network | No improvement |
| Throughput | 1 user | 1 user (thesis scope) | Unnecessary |
| Memory Usage | 16 GB RAM | 16 GB + DB overhead | Worse |
| Deployment Complexity | `streamlit run app.py` | Docker + DB + API + Frontend | Significantly worse |

**Conclusion**: Backend adds latency and complexity without performance benefits for single-user local execution.

### 5. Development Cost vs Research Value

| Component | Estimated Effort | Research Value |
|-----------|------------------|----------------|
| Hybrid Retrieval Algorithm | 40 hours | **High** (core contribution) |
| Semantic Chunking Implementation | 30 hours | **High** (core contribution) |
| RAGAS Evaluation Framework | 25 hours | **High** (core contribution) |
| **Backend API (FastAPI)** | **30-40 hours** | **Low** (infrastructure) |
| **Database Schema & ORM** | **20-30 hours** | **Low** (infrastructure) |
| **RBAC Implementation** | **15-20 hours** | **Medium** (requirement) |
| **Frontend-Backend Integration** | **15-20 hours** | **Low** (integration) |

**Total Backend Cost**: 80-110 hours (3-4 weeks)
**Alternative Streamlit Enhancement**: 20-30 hours (3-5 days)

**Verdict**: Backend is a poor investment of thesis development time.

### 6. Deployment & Maintenance

**Current Deployment**:
```bash
# Single command
streamlit run src/ui/app.py
```

**Backend Deployment**:
```bash
# Minimum 4 services
docker-compose up -d  # PostgreSQL
uvicorn main:app --reload  # FastAPI
streamlit run app.py  # Frontend
# Plus: Nginx, environment config, migrations
```

**Thesis Defense Reality**: Examiners care about **research methodology and results**, not whether you used microservices architecture.

---

## Optimal Alternative Approach: Enhanced Streamlit

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              ENHANCED STREAMLIT ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Streamlit Application Layer                  │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  Authentication Layer (st.login / custom session)  │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  RBAC Middleware (role-based UI rendering)          │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  RAG Pipeline (preprocessing, retrieval, generation)│  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  Evaluation Engine (Hit Rate, MRR, RAGAS)           │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         ▼                   ▼                   ▼               │
│  ┌───────────────┐  ┌─────────────────┐  ┌──────────────┐      │
│  │ Local Files   │  │ FAISS / Chroma  │  │ Ollama       │      │
│  │ (JSON/CSV)    │  │ (Vector DB)     │  │ (Local LLM)  │      │
│  └───────────────┘  └─────────────────┘  └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Phase 1: Lightweight Authentication (2-3 days)
```python
# users.json - Simple user store
{
  "researcher": {
    "password_hash": "scrypt:32768:8:1$...",
    "role": "researcher",
    "permissions": ["read", "write", "execute", "configure"]
  },
  "practitioner": {
    "password_hash": "scrypt:32768:8:1$...",
    "role": "practitioner",
    "permissions": ["read", "use"]
  }
}

# auth.py - Session management
def authenticate(username, password):
    user = load_user(username)
    if verify_password(password, user.password_hash):
        st.session_state.authenticated = True
        st.session_state.role = user.role
        st.session_state.permissions = user.permissions
        log_audit(username, "login")
        return True
    return False

def require_permission(permission):
    if permission not in st.session_state.get("permissions", []):
        st.error("Akses ditolak")
        st.stop()
```

#### Phase 2: RBAC UI Integration (1-2 days)
```python
# Conditional UI rendering based on role
if st.session_state.role == "researcher":
    with st.sidebar.expander("🔬 Peneliti Menu"):
        st.button("Upload Korpus")
        st.button("Konfigurasi Chunking")
        st.button("Jalankan Evaluasi")
elif st.session_state.role == "practitioner":
    with st.sidebar.expander("⚖️ Praktisi Menu"):
        st.button("Chat dengan Asisten Hukum")
        # No configuration access
```

#### Phase 3: Audit Logging (1 day)
```python
# audit_log.json - Append-only log
{
  "timestamp": "2026-08-13T14:30:00",
  "user": "researcher",
  "action": "configure_chunking",
  "parameters": {"threshold": 0.75},
  "ip_address": "127.0.0.1"
}
```

#### Phase 4: Configuration Persistence (1 day)
```python
# config.json - System configuration
{
  "chunking": {"threshold": 0.75, "model": "indobenchmark/indobert-base-p1"},
  "retrieval": {"top_k": 10, "rrf_k": 60},
  "llm": {"model": "llama3", "temperature": 0.0},
  "evaluation": {"golden_dataset": "golden_dataset_rag_hukum_indonesia_rev3.csv"}
}
```

### Total Additional Effort: 5-7 days

**Benefits**:
- ✅ Maintains focus on research contribution
- ✅ Adds required RBAC functionality
- ✅ Provides audit trail for thesis documentation
- ✅ Keeps deployment simple
- ✅ Easy to debug and modify

---

## Database.sql: Reference Artifact

While a dedicated backend is **not recommended** for the current thesis scope, I have generated a comprehensive `database.sql` file as a **reference artifact** for future scalability. This file documents the complete relational schema that would be required if the system were to be deployed as a multi-user web application after the thesis.

**Location**: `D:\Arsip Belajar di Itenas\Semester 7\Tugas Akhir\TA Rainova\File Hukum\database.sql`

**Contents**:
- Complete ERD with 8 tables
- Relational mappings (1:N, N:M)
- Constraints and indexes
- Sample data inserts
- Views for common queries

This file serves as:
1. **Documentation** of the data model requirements
2. **Future reference** if the system needs to be scaled
3. **Demonstration** of database design competence
4. **Appendix material** for the thesis

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Streamlit session state lost on refresh | Medium | Low | Use `st.cache_resource` for models, persist config to JSON |
| File-based storage corruption | Low | Medium | Regular backups, atomic writes |
| Concurrent access issues | Low | Low | Single-user thesis scope |
| Authentication bypass | Low | Medium | Implement proper password hashing (scrypt/argon2) |

---

## Final Recommendation

| Scenario | Recommendation | Rationale |
|----------|----------------|-----------|
| **Thesis Completion** | ❌ No Backend | Focus on research contribution, not infrastructure |
| **Local Demo** | ❌ No Backend | Streamlit is sufficient for single-user demo |
| **Department Deployment** | ⚠️ Enhanced Streamlit | Add auth + logging, still no backend needed |
| **Public Web App** | ✅ Backend Required | Multi-user, concurrent access, security |

**Action Items**:
1. ✅ Keep current Streamlit architecture
2. ✅ Add lightweight session-based authentication
3. ✅ Implement RBAC via session state
4. ✅ Add file-based audit logging
5. ✅ Persist configuration to JSON
6. 📄 Generate `database.sql` as future reference
7. ❌ Do NOT build dedicated backend for thesis scope

---

## Conclusion

A dedicated backend is an **architectural over-engineering** for this thesis project. The research contribution lies in the RAG algorithm evaluation, not in web service architecture. The optimal approach is to **enhance the existing Streamlit application** with lightweight authentication and file-based persistence, which adds necessary functionality with minimal complexity and development time.

The `database.sql` file is provided as a supplementary reference for future scalability, demonstrating forward-thinking design without compromising current thesis priorities.

---

**Assessment conducted by**: Kilo Engineering Assistant  
**Date**: 2026-08-13  
**Project**: RAG Asisten Regulasi KUHP Baru  
**Thesis**: Rainova Rahaniawan (152023007) - ITENAS
