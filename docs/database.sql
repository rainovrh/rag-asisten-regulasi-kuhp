-- ============================================================================
-- DATABASE SCHEMA: Sistem Asisten Regulasi RAG
-- KUHP Baru (UU No. 1 Tahun 2023) - Hybrid Retrieval & Semantic Chunking
-- ============================================================================
-- Generated for: Rainova Rahaniawan (152023007) - ITENAS
-- Purpose: Reference schema for future scalability beyond thesis scope
-- Note: This schema is NOT required for current thesis implementation
--       Current implementation uses file-based storage (JSON, FAISS, CSV)
--       Path references updated for modular src/ architecture (v1.0)
-- ============================================================================

-- ============================================================================
-- PATH REFERENCES (File-Based Storage)
-- ============================================================================
--
-- Corpus Data:      data/processed/kuhp_bersih.json
-- Vector Index:     data/indexes/faiss_index_kuhp/
-- Golden Dataset:   data/datasets/golden_dataset_rag_hukum_indonesia_rev3.csv
-- Evaluation Logs:  logs/evaluation_results.csv
-- Source Code:      src/{preprocessing,retrieval,generation,evaluation,ui,utils}/
--
-- ============================================================================

-- ============================================================================
-- EXTENSIONS
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================================
-- ENUMERATED TYPES
-- ============================================================================

CREATE TYPE user_role AS ENUM ('researcher', 'practitioner', 'administrator');
CREATE TYPE permission_level AS ENUM ('none', 'read', 'use', 'execute', 'configure', 'manage', 'full');
CREATE TYPE evaluation_status AS ENUM ('pending', 'running', 'completed', 'failed');
CREATE TYPE document_type AS ENUM ('kuhp', 'peraturan_pemerintah', 'peraturan_menteri', 'keputusan_presiden', 'lainnya');
CREATE TYPE chunking_method AS ENUM ('semantic', 'fixed_size', 'recursive', 'paragraph');
CREATE TYPE retrieval_method AS ENUM ('bm25', 'dense', 'hybrid', 'crag');
CREATE TYPE metric_type AS ENUM ('hit_rate', 'mrr', 'faithfulness', 'answer_relevance', 'precision', 'recall');

-- ============================================================================
-- TABLE DEFINITIONS
-- ============================================================================

-- ============================================================================
-- 1. USERS & AUTHENTICATION
-- ============================================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    role user_role NOT NULL DEFAULT 'practitioner',
    is_active BOOLEAN DEFAULT TRUE,
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID REFERENCES users(id),
    
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);

-- ============================================================================
-- 2. PERMISSIONS & RBAC
-- ============================================================================

CREATE TABLE permissions (
    id SERIAL PRIMARY KEY,
    code VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    module VARCHAR(50) NOT NULL,
    level permission_level NOT NULL
);

CREATE TABLE role_permissions (
    role user_role NOT NULL,
    permission_id INTEGER NOT NULL REFERENCES permissions(id),
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    granted_by UUID REFERENCES users(id),
    
    PRIMARY KEY (role, permission_id)
);

-- ============================================================================
-- 3. SYSTEM CONFIGURATION
-- ============================================================================

CREATE TABLE system_configs (
    id SERIAL PRIMARY KEY,
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by UUID REFERENCES users(id)
);

CREATE INDEX idx_system_configs_key ON system_configs(config_key);
CREATE INDEX idx_system_configs_active ON system_configs(is_active);

-- ============================================================================
-- 4. DOCUMENT REPOSITORY
-- ============================================================================

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    document_type document_type NOT NULL,
    source_url TEXT,
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    page_count INTEGER,
    is_indexed BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    uploaded_by UUID REFERENCES users(id),
    
    CONSTRAINT valid_file_hash CHECK (length(file_hash) = 64)
);

CREATE INDEX idx_documents_type ON documents(document_type);
CREATE INDEX idx_documents_indexed ON documents(is_indexed);
CREATE INDEX idx_documents_created_at ON documents(created_at);

-- ============================================================================
-- 5. CHUNKS (SEMANTIC UNITS)
-- ============================================================================

CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    pasal_reference VARCHAR(100),
    ayat_reference VARCHAR(100),
    chunking_method chunking_method NOT NULL,
    token_count INTEGER,
    char_count INTEGER NOT NULL,
    embedding_vector VECTOR(768), -- For indobenchmark/indobert-base-p1
    embedding_model VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(document_id, chunk_index)
);

CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_pasal ON chunks(pasal_reference);
CREATE INDEX idx_chunks_method ON chunks(chunking_method);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding_vector vector_cosine_ops);

-- ============================================================================
-- 6. VECTOR DATABASES
-- ============================================================================

CREATE TABLE vector_databases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    db_type VARCHAR(50) NOT NULL, -- 'faiss', 'chroma', 'pinecone', 'weaviate'
    storage_path TEXT NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,
    dimension INTEGER NOT NULL,
    index_parameters JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID REFERENCES users(id),
    
    CONSTRAINT valid_dimension CHECK (dimension > 0)
);

CREATE INDEX idx_vector_databases_type ON vector_databases(db_type);
CREATE INDEX idx_vector_databases_active ON vector_databases(is_active);

-- ============================================================================
-- 7. EVALUATION DATASETS (GOLDEN DATASET)
-- ============================================================================

CREATE TABLE evaluation_datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    version VARCHAR(20) NOT NULL,
    total_scenarios INTEGER NOT NULL,
    is_verified BOOLEAN DEFAULT FALSE,
    verified_by UUID REFERENCES users(id),
    verified_at TIMESTAMP WITH TIME ZONE,
    file_path TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_version CHECK (version ~* '^v?\d+\.\d+$')
);

CREATE INDEX idx_evaluation_datasets_version ON evaluation_datasets(version);
CREATE INDEX idx_evaluation_datasets_verified ON evaluation_datasets(is_verified);

-- ============================================================================
-- 8. EVALUATION SCENARIOS
-- ============================================================================

CREATE TABLE evaluation_scenarios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES evaluation_datasets(id) ON DELETE CASCADE,
    scenario_number INTEGER NOT NULL,
    query TEXT NOT NULL,
    context_type VARCHAR(50), -- 'factoid', 'open-ended', 'comparative'
    expected_pasal VARCHAR(100) NOT NULL,
    ground_truth TEXT NOT NULL,
    difficulty_level VARCHAR(20), -- 'easy', 'medium', 'hard'
    requires_cross_article BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(dataset_id, scenario_number)
);

CREATE INDEX idx_evaluation_scenarios_dataset ON evaluation_scenarios(dataset_id);
CREATE INDEX idx_evaluation_scenarios_type ON evaluation_scenarios(context_type);
CREATE INDEX idx_evaluation_scenarios_difficulty ON evaluation_scenarios(difficulty_level);
CREATE INDEX idx_evaluation_scenarios_cross_article ON evaluation_scenarios(requires_cross_article);

-- ============================================================================
-- 9. EVALUATION RUNS
-- ============================================================================

CREATE TABLE evaluation_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id UUID NOT NULL REFERENCES evaluation_datasets(id),
    run_name VARCHAR(100) NOT NULL,
    description TEXT,
    config JSONB NOT NULL, -- chunking_method, retrieval_method, model params
    status evaluation_status DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    total_scenarios INTEGER NOT NULL,
    completed_scenarios INTEGER DEFAULT 0,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_status CHECK (
        (status = 'pending' AND started_at IS NULL) OR
        (status = 'running' AND started_at IS NOT NULL AND completed_at IS NULL) OR
        (status IN ('completed', 'failed') AND started_at IS NOT NULL AND completed_at IS NOT NULL)
    )
);

CREATE INDEX idx_evaluation_runs_dataset ON evaluation_runs(dataset_id);
CREATE INDEX idx_evaluation_runs_status ON evaluation_runs(status);
CREATE INDEX idx_evaluation_runs_created_by ON evaluation_runs(created_by);
CREATE INDEX idx_evaluation_runs_created_at ON evaluation_runs(created_at);

-- ============================================================================
-- 10. EVALUATION RESULTS (INDIVIDUAL SCENARIOS)
-- ============================================================================

CREATE TABLE evaluation_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    scenario_id UUID NOT NULL REFERENCES evaluation_scenarios(id),
    
    -- Query & Generated Answer
    query TEXT NOT NULL,
    generated_answer TEXT NOT NULL,
    ground_truth TEXT NOT NULL,
    
    -- Retrieval Metrics
    hit BOOLEAN NOT NULL,
    first_relevant_rank INTEGER,
    hit_at_k BOOLEAN NOT NULL,
    reciprocal_rank FLOAT,
    
    -- RAGAS Metrics
    faithfulness_score FLOAT CHECK (faithfulness_score >= 0 AND faithfulness_score <= 1),
    answer_relevance_score FLOAT CHECK (answer_relevance_score >= 0 AND answer_relevance_score <= 1),
    context_precision FLOAT CHECK (context_precision >= 0 AND context_precision <= 1),
    context_recall FLOAT CHECK (context_recall >= 0 AND context_recall <= 1),
    
    -- Retrieved Contexts (JSON array)
    retrieved_chunks JSONB,
    chunk_scores JSONB,
    
    -- Additional Metadata
    latency_retrieval_ms INTEGER,
    latency_generation_ms INTEGER,
    total_latency_ms INTEGER,
    error_message TEXT,
    metadata JSONB,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(run_id, scenario_id)
);

CREATE INDEX idx_evaluation_results_run ON evaluation_results(run_id);
CREATE INDEX idx_evaluation_results_scenario ON evaluation_results(scenario_id);
CREATE INDEX idx_evaluation_results_faithfulness ON evaluation_results(faithfulness_score);
CREATE INDEX idx_evaluation_results_relevance ON evaluation_results(answer_relevance_score);

-- ============================================================================
-- 11. AGGREGATED EVALUATION METRICS
-- ============================================================================

CREATE TABLE evaluation_aggregates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id UUID NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    
    -- Retrieval Metrics
    hit_rate FLOAT NOT NULL CHECK (hit_rate >= 0 AND hit_rate <= 1),
    hit_rate_at_k FLOAT NOT NULL CHECK (hit_rate_at_k >= 0 AND hit_rate_at_k <= 1),
    mrr FLOAT NOT NULL CHECK (mrr >= 0 AND mrr <= 1),
    mean_first_rank FLOAT,
    
    -- RAGAS Metrics
    avg_faithfulness FLOAT NOT NULL CHECK (avg_faithfulness >= 0 AND avg_faithfulness <= 1),
    avg_answer_relevance FLOAT NOT NULL CHECK (avg_answer_relevance >= 0 AND avg_answer_relevance <= 1),
    avg_context_precision FLOAT CHECK (avg_context_precision >= 0 AND avg_context_precision <= 1),
    avg_context_recall FLOAT CHECK (avg_context_recall >= 0 AND avg_context_recall <= 1),
    
    -- Statistical Measures
    std_faithfulness FLOAT,
    std_answer_relevance FLOAT,
    min_faithfulness FLOAT,
    max_faithfulness FLOAT,
    min_answer_relevance FLOAT,
    max_answer_relevance FLOAT,
    
    -- Performance Metrics
    avg_latency_retrieval_ms FLOAT,
    avg_latency_generation_ms FLOAT,
    avg_total_latency_ms FLOAT,
    
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(run_id)
);

CREATE INDEX idx_evaluation_aggregates_run ON evaluation_aggregates(run_id);

-- ============================================================================
-- 12. AUDIT LOGS
-- ============================================================================

CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_action CHECK (action ~* '^[a-z_]+$')
);

CREATE INDEX idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource_type, resource_id);

-- ============================================================================
-- 13. SYSTEM SESSIONS
-- ============================================================================

CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_token ON sessions(session_token);
CREATE INDEX idx_sessions_expires ON sessions(expires_at);
CREATE INDEX idx_sessions_active ON sessions(is_active);

-- ============================================================================
-- 14. MODEL REGISTRY
-- ============================================================================

CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'embedding', 'llm', 'reranker'
    provider VARCHAR(50) NOT NULL,
    model_id VARCHAR(200) NOT NULL,
    version VARCHAR(50),
    parameters JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(model_type, model_id, version)
);

CREATE INDEX idx_models_type ON models(model_type);
CREATE INDEX idx_models_provider ON models(provider);
CREATE INDEX idx_models_active ON models(is_active);

-- ============================================================================
-- 15. CHUNKING EXPERIMENTS
-- ============================================================================

CREATE TABLE chunking_experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    chunking_method chunking_method NOT NULL,
    parameters JSONB NOT NULL, -- threshold, max_tokens, overlap, etc.
    document_ids UUID[] NOT NULL,
    total_chunks INTEGER NOT NULL,
    avg_chunk_size FLOAT,
    storage_path TEXT,
    is_baseline BOOLEAN DEFAULT FALSE,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_chunking_experiments_method ON chunking_experiments(chunking_method);
CREATE INDEX idx_chunking_experiments_baseline ON chunking_experiments(is_baseline);

-- ============================================================================
-- 16. RETRIEVAL EXPERIMENTS
-- ============================================================================

CREATE TABLE retrieval_experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    retrieval_method retrieval_method NOT NULL,
    parameters JSONB NOT NULL, -- top_k, rrf_k, crag_threshold, etc.
    chunking_experiment_id UUID REFERENCES chunking_experiments(id),
    vector_db_id UUID REFERENCES vector_databases(id),
    is_baseline BOOLEAN DEFAULT FALSE,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_retrieval_experiments_method ON retrieval_experiments(retrieval_method);
CREATE INDEX idx_retrieval_experiments_baseline ON retrieval_experiments(is_baseline);

-- ============================================================================
-- 17. CHAT HISTORY
-- ============================================================================

CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    session_name VARCHAR(100),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB
);

CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    retrieved_chunks JSONB,
    latency_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_role CHECK (role IN ('user', 'assistant', 'system'))
);

CREATE INDEX idx_chat_sessions_user ON chat_sessions(user_id);
CREATE INDEX idx_chat_sessions_active ON chat_sessions(is_active);
CREATE INDEX idx_chat_messages_session ON chat_messages(session_id);
CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at);

-- ============================================================================
-- FOREIGN KEY CONSTRAINTS (Additional)
-- ============================================================================

ALTER TABLE documents ADD CONSTRAINT fk_documents_uploaded_by 
    FOREIGN KEY (uploaded_by) REFERENCES users(id) ON DELETE SET NULL;

ALTER TABLE evaluation_datasets ADD CONSTRAINT fk_evaluation_datasets_verified_by 
    FOREIGN KEY (verified_by) REFERENCES users(id) ON DELETE SET NULL;

ALTER TABLE evaluation_runs ADD CONSTRAINT fk_evaluation_runs_created_by 
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL;

ALTER TABLE chunking_experiments ADD CONSTRAINT fk_chunking_experiments_created_by 
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL;

ALTER TABLE retrieval_experiments ADD CONSTRAINT fk_retrieval_experiments_created_by 
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL;

ALTER TABLE chat_sessions ADD CONSTRAINT fk_chat_sessions_user_id 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- View: Latest evaluation run summary
CREATE VIEW v_latest_evaluation_summary AS
SELECT 
    er.id AS run_id,
    er.run_name,
    er.status,
    er.started_at,
    er.completed_at,
    ed.name AS dataset_name,
    u.username AS created_by,
    ea.hit_rate,
    ea.mrr,
    ea.avg_faithfulness,
    ea.avg_answer_relevance,
    EXTRACT(EPOCH FROM (er.completed_at - er.started_at)) AS duration_seconds
FROM evaluation_runs er
JOIN evaluation_datasets ed ON er.dataset_id = ed.id
JOIN users u ON er.created_by = u.id
LEFT JOIN evaluation_aggregates ea ON er.id = ea.run_id
WHERE er.created_at = (
    SELECT MAX(created_at) 
    FROM evaluation_runs 
    WHERE dataset_id = er.dataset_id
);

-- View: User activity summary
CREATE VIEW v_user_activity_summary AS
SELECT 
    u.id AS user_id,
    u.username,
    u.role,
    COUNT(DISTINCT er.id) AS total_evaluations,
    COUNT(DISTINCT cs.id) AS total_chat_sessions,
    MAX(al.timestamp) AS last_activity
FROM users u
LEFT JOIN evaluation_runs er ON u.id = er.created_by
LEFT JOIN chat_sessions cs ON u.id = cs.user_id
LEFT JOIN audit_logs al ON u.id = al.user_id
GROUP BY u.id, u.username, u.role;

-- View: Document usage statistics
CREATE VIEW v_document_usage_stats AS
SELECT 
    d.id AS document_id,
    d.title,
    d.document_type,
    COUNT(DISTINCT c.id) AS total_chunks,
    COUNT(DISTINCT er.id) AS times_used_in_evaluation,
    MAX(er.created_at) AS last_used_at
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
LEFT JOIN evaluation_results er2 ON TRUE -- Simplified join
LEFT JOIN evaluation_runs er ON er.id = er2.run_id
GROUP BY d.id, d.title, d.document_type;

-- ============================================================================
-- STORED PROCEDURES
-- ============================================================================

-- Procedure: Calculate aggregate metrics for an evaluation run
CREATE OR REPLACE FUNCTION calculate_evaluation_aggregates(p_run_id UUID)
RETURNS VOID AS $$
BEGIN
    INSERT INTO evaluation_aggregates (
        run_id,
        hit_rate,
        hit_rate_at_k,
        mrr,
        avg_faithfulness,
        avg_answer_relevance,
        avg_context_precision,
        avg_context_recall,
        std_faithfulness,
        std_answer_relevance,
        min_faithfulness,
        max_faithfulness,
        min_answer_relevance,
        max_answer_relevance,
        avg_latency_retrieval_ms,
        avg_latency_generation_ms,
        avg_total_latency_ms
    )
    SELECT
        p_run_id,
        AVG(CASE WHEN hit THEN 1.0 ELSE 0.0 END),
        AVG(CASE WHEN hit_at_k THEN 1.0 ELSE 0.0 END),
        AVG(reciprocal_rank),
        AVG(faithfulness_score),
        AVG(answer_relevance_score),
        AVG(context_precision),
        AVG(context_recall),
        STDDEV(faithfulness_score),
        STDDEV(answer_relevance_score),
        MIN(faithfulness_score),
        MAX(faithfulness_score),
        MIN(answer_relevance_score),
        MAX(answer_relevance_score),
        AVG(latency_retrieval_ms),
        AVG(latency_generation_ms),
        AVG(total_latency_ms)
    FROM evaluation_results
    WHERE run_id = p_run_id;
    
    UPDATE evaluation_runs
    SET status = 'completed',
        completed_at = NOW(),
        completed_scenarios = (SELECT COUNT(*) FROM evaluation_results WHERE run_id = p_run_id)
    WHERE id = p_run_id;
END;
$$ LANGUAGE plpgsql;

-- Procedure: Clean up expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM sessions
    WHERE expires_at < NOW() OR (is_active = FALSE AND last_accessed_at < NOW() - INTERVAL '7 days');
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Trigger: Update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_configs_updated_at BEFORE UPDATE ON system_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger: Auto-log configuration changes
CREATE OR REPLACE FUNCTION log_config_changes()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_logs (action, resource_type, resource_id, old_values, new_values)
    VALUES (
        'update_config',
        'system_config',
        OLD.id,
        row_to_json(OLD),
        row_to_json(NEW)
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER log_system_configs_changes AFTER UPDATE ON system_configs
    FOR EACH ROW EXECUTE FUNCTION log_config_changes();

-- ============================================================================
-- INITIAL DATA SEED
-- ============================================================================

-- Seed Permissions
INSERT INTO permissions (code, name, description, module, level) VALUES
('kb_read', 'Membaca Basis Pengetahuan', 'Melihat daftar dokumen dan pasal', 'knowledge_base', 'read'),
('kb_write', 'Mengunggah Dokumen', 'Upload dokumen regulasi baru', 'knowledge_base', 'write'),
('kb_execute', 'Memproses Indeksasi', 'Menjalankan preprocessing dan chunking', 'knowledge_base', 'execute'),
('kb_configure', 'Mengonfigurasi Chunking', 'Mengubah parameter semantic chunking', 'knowledge_base', 'configure'),
('retrieval_execute', 'Menjalankan Retrieval', 'Mencari pasal menggunakan hybrid retrieval', 'retrieval', 'execute'),
('retrieval_configure', 'Mengonfigurasi Retrieval', 'Mengubah parameter BM25, Dense, RRF', 'retrieval', 'configure'),
('eval_execute', 'Menjalankan Evaluasi', 'Menjalankan batch evaluation', 'evaluation', 'execute'),
('eval_read', 'Melihat Hasil Evaluasi', 'Melihat laporan evaluasi', 'evaluation', 'read'),
('eval_configure', 'Mengonfigurasi Evaluasi', 'Mengunggah dataset dan mengatur metrik', 'evaluation', 'configure'),
('chat_use', 'Menggunakan Chat', 'Berinteraksi dengan asisten hukum', 'chat', 'use'),
('admin_manage', 'Manajemen Sistem', 'Mengelola pengguna dan konfigurasi sistem', 'administration', 'manage')
ON CONFLICT (code) DO NOTHING;

-- Seed Role Permissions
INSERT INTO role_permissions (role, permission_id, granted_by)
SELECT 'researcher', id, NULL FROM permissions WHERE code IN (
    'kb_read', 'kb_write', 'kb_execute', 'kb_configure',
    'retrieval_execute', 'retrieval_configure',
    'eval_execute', 'eval_read', 'eval_configure',
    'chat_use'
)
ON CONFLICT DO NOTHING;

INSERT INTO role_permissions (role, permission_id, granted_by)
SELECT 'practitioner', id, NULL FROM permissions WHERE code IN (
    'kb_read', 'retrieval_execute', 'chat_use'
)
ON CONFLICT DO NOTHING;

INSERT INTO role_permissions (role, permission_id, granted_by)
SELECT 'administrator', id, NULL FROM permissions
ON CONFLICT DO NOTHING;

-- Seed Default System Configuration
INSERT INTO system_configs (config_key, config_value, description)
VALUES (
    'default_preprocessing',
    '{"remove_page_numbers": true, "merge_broken_sentences": true, "lowercase": true, "expand_abbreviations": {"uu": "undang-undang", "kuhp": "kitab undang-undang hukum pidana"}}',
    'Default preprocessing configuration'
),
(
    'default_semantic_chunking',
    '{"enabled": true, "threshold": 0.75, "embedding_model": "indobenchmark/indobert-base-p1"}',
    'Default semantic chunking configuration'
),
(
    'default_hybrid_retrieval',
    '{"sparse_top_k": 10, "dense_top_k": 10, "rrf_k": 60, "final_top_k": 5}',
    'Default hybrid retrieval configuration'
),
(
    'default_llm',
    '{"model": "meta-llama/Meta-Llama-3-8B-Instruct", "quantization": "4-bit", "temperature": 0.1, "max_tokens": 512}',
    'Default LLM configuration'
)
ON CONFLICT (config_key) DO NOTHING;

-- Seed Default Model Registry
INSERT INTO models (name, model_type, provider, model_id, version, is_active)
VALUES
('IndoBERT Base', 'embedding', 'huggingface', 'indobenchmark/indobert-base-p1', 'v1', TRUE),
('BGE M3', 'embedding', 'huggingface', 'BAAI/bge-m3', 'v1', FALSE),
('Llama-3 8B', 'llm', 'ollama', 'llama3', '8b', TRUE),
('MiniLM L12', 'embedding', 'huggingface', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'v1', FALSE)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- FUNCTIONS FOR FULL-TEXT SEARCH
-- ============================================================================

-- Function: Search chunks by similarity
CREATE OR REPLACE FUNCTION search_chunks_similar(
    query_embedding VECTOR(768),
    match_threshold FLOAT DEFAULT 0.7,
    match_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    chunk_text TEXT,
    pasal_reference VARCHAR(100),
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.chunk_text,
        c.pasal_reference,
        1 - (c.embedding_vector <=> query_embedding) AS similarity
    FROM chunks c
    WHERE c.embedding_vector IS NOT NULL
    AND 1 - (c.embedding_vector <=> query_embedding) > match_threshold
    ORDER BY c.embedding_vector <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function: Search chunks by BM25-like text search
CREATE OR REPLACE FUNCTION search_chunks_text(
    search_query TEXT,
    match_count INTEGER DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    chunk_text TEXT,
    pasal_reference VARCHAR(100),
    rank FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.chunk_text,
        c.pasal_reference,
        ts_rank(to_tsvector('indonesian', c.chunk_text), websearch_to_tsquery('indonesian', search_query)) AS rank
    FROM chunks c
    WHERE to_tsvector('indonesian', c.chunk_text) @@ websearch_to_tsquery('indonesian', search_query)
    ORDER BY rank DESC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- GIN index for full-text search on chunks
CREATE INDEX IF NOT EXISTS idx_chunks_text_search ON chunks USING GIN (to_tsvector('indonesian', chunk_text));

-- B-tree index for pasal reference (commonly queried)
CREATE INDEX IF NOT EXISTS idx_chunks_pasal_btree ON chunks(pasal_reference);

-- Composite index for evaluation results queries
CREATE INDEX IF NOT EXISTS idx_evaluation_results_run_faithfulness 
    ON evaluation_results(run_id, faithfulness_score);

-- BRIN index for time-series data (audit_logs, chat_messages)
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp_brin ON audit_logs USING BRIN(timestamp);
CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp_brin ON chat_messages USING BRIN(created_at);

-- ============================================================================
-- MAINTENANCE
-- ============================================================================

-- Vacuum and analyze schedule (requires pg_cron extension)
-- SELECT cron.schedule('vacuum_analyze', '0 2 * * *', 'VACUUM ANALYZE');

-- ============================================================================
-- SCHEMA DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE users IS 'System users with RBAC roles';
COMMENT ON TABLE permissions IS 'Available permissions in the system';
COMMENT ON TABLE role_permissions IS 'Role-permission mapping';
COMMENT ON TABLE system_configs IS 'System configuration key-value store';
COMMENT ON TABLE documents IS 'Uploaded legal documents';
COMMENT ON TABLE chunks IS 'Semantic chunks extracted from documents';
COMMENT ON TABLE vector_databases IS 'Vector database configurations';
COMMENT ON TABLE evaluation_datasets IS 'Golden datasets for evaluation';
COMMENT ON TABLE evaluation_scenarios IS 'Individual evaluation scenarios';
COMMENT ON TABLE evaluation_runs IS 'Evaluation execution runs';
COMMENT ON TABLE evaluation_results IS 'Individual scenario results';
COMMENT ON TABLE evaluation_aggregates IS 'Aggregated metrics per run';
COMMENT ON TABLE audit_logs IS 'System audit trail';
COMMENT ON TABLE sessions IS 'User sessions';
COMMENT ON TABLE models IS 'ML model registry';
COMMENT ON TABLE chunking_experiments IS 'Chunking method experiments';
COMMENT ON TABLE retrieval_experiments IS 'Retrieval method experiments';
COMMENT ON TABLE chat_sessions IS 'User chat sessions';
COMMENT ON TABLE chat_messages IS 'Individual chat messages';

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
