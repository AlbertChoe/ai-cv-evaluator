# AI CV & Project Evaluator

End-to-end FastAPI service that ingests hiring artifacts, evaluates candidate submissions with Retrieval-Augmented Generation (RAG), and surfaces asynchronous scoring results backed by SQLite and Qdrant.

---

## Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Operational Flow](#operational-flow)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. API Evaluation Lifecycle](#2-api-evaluation-lifecycle)
  - [3. Evaluation Pipeline (LLM Chain)](#3-evaluation-pipeline-llm-chain)
- [HTTP API Reference](#http-api-reference)
- [Persistence & Storage](#persistence--storage)
- [Retrieval-Augmented Generation](#retrieval-augmented-generation)
- [LLM Integration & Hardening](#llm-integration--hardening)
- [Logging, Monitoring, and Health Checks](#logging-monitoring-and-health-checks)

---

## Key Features

- **FastAPI service** with layered modules (`api`, `domain`, `infra`) and Pydantic schemas.
- **Asynchronous evaluation jobs**: `/evaluate` queues background work; `/result/{id}` reads status/results.
- **Persistent storage** via SQLite (metadata, job status/results) and local disk for uploaded PDFs.
- **Vector search (Qdrant)** for job descriptions, rubrics, and case briefs; automatic collection/index setup.
- **RAG-driven LLM evaluation** combining JD, rubric, and case brief context before scoring CVs and reports.
- **Multi-stage LLM chain** with strict JSON output validation and exponential backoff retries for OpenAI/OpenRouter.
- **Health observability** including `/vector-db/health` and structured evaluation logs (`evaluation_debug.log`).

---

## Architecture Overview

```
app/
├─ main.py                # FastAPI app bootstrap
├─ settings.py
├─ logging.py, error_handlers.py
api/
├─ router.py              # Top-level router
└─ endpoints/             # upload, evaluate, result, health
domain/
├─ schemas.py             # DTOs
└─ services/evaluation_pipeline.py
infra/
├─ db/                    # SQLAlchemy engine, models, session
├─ llm/                   # Prompt templates + client (OpenAI/OpenRouter)
├─ rag/                   # Qdrant client + retrieval helpers
└─ repositories/          # FilesRepository, JobsRepository
ingest/
└─ ingest_all.py          # Combined JD/Brief/Rubric ingestion
```

- **API layer** exposes HTTP endpoints, validates requests, and orchestrates repositories/services.
- **Domain layer** houses business workflows (`evaluation_pipeline`) and shared schemas.
- **Infrastructure layer** handles persistence (SQLite), file storage, vector DB access, and LLM integrations.
- **Asynchronous execution** uses `asyncio.create_task` to decouple long-running evaluations from request latency.

---

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# Start Qdrant (Docker)
docker compose up -d qdrant

# Launch API
uvicorn app.main:app --reload --host 127.0.0.1 --port 8008
# Docs: http://127.0.0.1:8008/docs
```

Populate Qdrant with role-specific artifacts (JD, case brief, rubric) using the ingestion script (see [Data Ingestion](#1-data-ingestion)).

---

## Configuration

All environment variables are loaded via `app/settings.py`. Key options:

| Variable                | Default                         | Description                                       |
|-------------------------|---------------------------------|---------------------------------------------------|
| `APP_NAME`              | AI CV & Project Evaluator       | FastAPI title                                     |
| `LOG_LEVEL`             | INFO                            | Global log level                                  |
| `STORAGE_DIR`           | `storage`                       | Disk location for uploaded PDFs                   |
| `SQLITE_PATH`           | `app.sqlite3`                   | SQLite DB file path                               |
| `QDRANT_URL`            | `http://localhost:6333`         | Qdrant endpoint                                   |
| `QDRANT_API_KEY`        | *(empty)*                       | Optional Qdrant auth                              |
| `OPENAI_API_KEY`        | *(required for production)*     | OpenAI key for chat + embeddings                  |
| `OPENAI_MODEL`          | `gpt-4o-mini`                   | Chat model for evaluations                        |
| `OPENAI_EMBEDDING_MODEL`| `text-embedding-3-small`        | Embedding model for RAG vectors                   |
| `OPENROUTER_API_KEY`    | *(optional)*                    | Alternative LLM provider                          |
| `OPENROUTER_MODEL`      | `openai/gpt-4o-mini`            | OpenRouter model slug                             |

---

## Operational Flow

### 1. Data Ingestion

Use `ingest/ingest_all.py` to ingest job assets prior to evaluation:

```bash
python -m ingest.ingest_all \
  --jd data/Frontend_Engineer_JobDesc.pdf \
  --brief data/case_study_brief_backend.pdf \
  --rubric data/Frontend_Engineer_Rubric.pdf
```

Steps executed:
1. **Catalog metadata**: First two JD pages summarized via LLM to create standardized title, aliases, tags, and `job_key` (`ingest_all.upsert_catalog`).
2. **JD chunking**: Full JD is chunked (size 1000, overlap 150), embedded, and upserted into Qdrant (`doc_type="jd_chunk"`).
3. **Case brief ingestion**: Brief is chunked similarly and stored (`doc_type="case_brief"`).
4. **Rubric parsing**: Tables are normalized into Markdown, chunked, and embedded (`doc_type="rubric"`).
5. Payload indexes (job_key, doc_type, etc.) are auto-created for efficient filtering (`infra/rag/qdrant_client.ensure_collection`).

Artifacts are keyed by `job_key` allowing the evaluation pipeline to fetch aligned references later.

### 2. API Evaluation Lifecycle

1. **Upload files** (`POST /upload`): Accepts CV and/or Project Report PDFs, writes to disk (`storage/`), records metadata in SQLite `files` table, and returns generated IDs.
2. **Trigger evaluation** (`POST /evaluate`): Validates file IDs, creates a job row (`status="queued"`), and schedules background evaluation with `asyncio.create_task`. Immediate response includes `job_id` and status.
3. **Poll results** (`GET /result/{job_id}`): Returns current job status (`queued`, `processing`, `completed`, `failed`). Once completed, includes RAG-backed scores and feedback.
4. **Health checks** (`GET /vector-db/health`): Validates Qdrant connectivity, returning available collections and counts.

### 3. Evaluation Pipeline (LLM Chain)

Located in `domain/services/evaluation_pipeline.py`:

1. **Job key resolution**: Finds the best-matching `job_key` in Qdrant catalog using embeddings and alias search (`infra/rag/retriever.resolve_job_key`).
2. **Document parsing**: PDFs are converted to text with `pdfplumber` (`infra/pdf/parser.py`).
3. **Reference retrieval**:
   - Shared rubric blocks fetched once (`retrieve_rubrics`).
   - CV references combine JD chunks + rubric context (`retrieve_for_cv`).
   - Project references combine case brief chunks + rubric context (`retrieve_for_project`).
   - Neighbor stitching merges adjacent vector hits for coherent context (`infra/rag/retriever._stitch`).
4. **LLM calls (three-stage chain)**:
   - `evaluate_cv_llm`: Compares CV text vs JD/rubric references.
   - `evaluate_project_llm`: Compares project report vs case brief/rubric references.
   - `summarize_overall_llm`: Synthesizes final recommendation using prior JSON outputs.
5. **Result persistence**: Numeric scores and stringified feedback stored in `job_results` table; status updated to `completed`. Errors capture exception messages with `status="failed"`.
6. **Logging**: Detailed trace (job key, retrieval counts, score previews) appended to `evaluation_debug.log` for diagnostics.

---

## HTTP API Reference

| Method | Path                 | Description | Request Highlights | Response |
|--------|----------------------|-------------|--------------------|----------|
| `POST` | `/upload`            | Store candidate files | Multipart form with `cv` and/or `report` PDFs | `UploadResponse` containing `cv_id` / `report_id` |
| `POST` | `/evaluate`          | Queue evaluation job  | JSON: `{ job_title, cv_id, report_id }` | `JobStatusResponse { id, status="queued" }` |
| `GET`  | `/result/{job_id}`   | Retrieve job status & result | URL param `job_id` | `JobStatusResponse` including `result` or `error` |
| `GET`  | `/vector-db/health`  | Qdrant health check   | – | `{ status, collections, collection_count }` |

Example `POST /evaluate` payload:
```json
{
  "job_title": "Frontend Engineer",
  "cv_id": "file_123",
  "report_id": "file_456"
}
```

Result payload (successful job):
```json
{
  "id": "job_abcd",
  "status": "completed",
  "result": {
    "cv_match_rate": 0.82,
    "cv_feedback": "- Solid React + TypeScript alignment\n- Clear evidence of API integration experience",
    "project_score": 4.5,
    "project_feedback": "- Addresses chaining requirements\n- Missing explicit error handling discussion",
    "overall_summary": "CV aligns strongly..."
  },
  "error": null
}
```

---

## Persistence & Storage

- **Files**: Uploaded PDFs stored under `STORAGE_DIR` with sanitized filenames. Records persisted in `files` table (`FilesRepository.save`).
- **Jobs**: `jobs` table tracks status, job title, and references to CV/report file IDs.
- **Results**: `job_results` table maintains scores and textual feedback for successful evaluations or error messages for failures.
- **Vector DB (Qdrant)**: Collections `job_catalog`, `job_descriptions`, and `case_and_rubrics` store embeddings keyed by `job_key`.
- **Initialization**: `init_db()` runs on FastAPI startup to ensure tables exist (`app/main.py:12-15`). Qdrant indexes instantiated lazily on demand (`infra/rag/qdrant_client.ensure_collection`).

---

## Retrieval-Augmented Generation

- **Embeddings**: OpenAI `text-embedding-3-small` used for catalog, JD, rubric, and case brief documents (see `infra/rag/embeddings.py`).
- **Vector search**: `search_top_k_filtered` filters by `job_key` and `doc_type` ensuring role-aligned retrieval. `fetch_neighbors_by_index` gathers sequential chunks to provide contiguous context.
- **Reference composition**:
  - CV evaluation: `[JD chunk(s)] + [rubric chunk(s)]`.
  - Project evaluation: `[case brief chunk(s)] + [rubric chunk(s)]`.
- **Safety**: Numeric samples scrubbed from references to prevent bias before prompt injection (`evaluation_pipeline.redact_numeric_examples`).

---

## LLM Integration & Hardening

- **Prompts** (`infra/llm/prompts.py`):
  - CV prompt asks for direct CV-to-JD comparison with labeled evidence quotes and fallback scoring rules.
  - Project prompt compares project report to case brief/rubric with similar guardrails.
  - Summary prompt enforces 3–5 sentence structured recommendation referencing exact metrics.
  - Catalog prompt standardizes job metadata during ingestion.
- **Retry/backoff**: `_post_with_retries` handles network/HTTP issues (5xx, 429, 408), doubling backoff per attempt (`infra/llm/client.py`).
- **Provider selection**: Prefers OpenAI when `OPENAI_API_KEY` is set; falls back to OpenRouter if configured; raises runtime error when neither available (preventing silent stub usage in production).

---

## Logging, Monitoring, and Health Checks

- **Structured evaluation logs**: `evaluation_debug.log` captures every stage (resolution, retrieval counts, score previews, summary text) for traceability.
- **FastAPI logging**: Configured via `app/logging.py` (rotating or level adjustments can be added there).
- **Vector DB health**: `/vector-db/health` confirms Qdrant availability and enumerates collections for quick diagnostics.

---

