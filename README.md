# AI CV & Project Evaluator — FastAPI + SQLite + Qdrant

- FastAPI API (layered)
- SQLite via SQLAlchemy for metadata/jobs
- Qdrant as Vector DB (Docker Compose)
- Local storage for PDFs
- Pluggable LLM/Embeddings (OpenAI/OpenRouter)

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d qdrant
uvicorn app.main:app --reload --host 127.0.0.1 --port 8008
# http://127.0.0.1:8000/docs
```

## Ingest
```bash
python -m ingest.ingest_job_desc
python -m ingest.ingest_case_brief
python -m ingest.ingest_rubrics


python -m ingest.ingest_all --jd data/job_description_backend.pdf --brief data/case_study_brief_backend.pdf --rubric data/scoring_rubric_backend.pdf

```