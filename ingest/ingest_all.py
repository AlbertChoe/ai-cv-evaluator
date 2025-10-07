import os
import re
import uuid
import asyncio
import pdfplumber
import logging
from typing import List
from infra.rag.embeddings import embed_texts_openai
from infra.rag.qdrant_client import (
    COLLECTION_CATALOG, ensure_collection, upsert_points_batch, upsert_texts_with_ids,
    COLLECTION_CV, COLLECTION_PROJECT
)
from infra.llm.client import generate_job_catalog_metadata

VECTOR_SIZE = 1536
HEADER_CELLS = {"parameter", "description", "scoring guide"}

log = logging.getLogger("ingest_all")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
for noisy_logger in ("httpx", "httpcore.httpx", "qdrant_client.http"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def read_pdf_text(path: str, max_pages: int | None = None) -> str:
    parts: List[str] = []
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages if max_pages is None else pdf.pages[:max_pages]
        for p in pages:
            parts.append(p.extract_text() or "")
    text = "\n".join(parts)
    return re.sub(r"\s+\n", "\n", text)


def chunk_text(text: str, size=1000, overlap=150) -> List[str]:
    out, i = [], 0
    n = len(text)
    while i < n:
        piece = text[i:i+size].strip()
        if piece:
            out.append(piece)
        i += max(1, size - overlap)
    return out


# helpers for rubrics pdf table
def is_header_row(row):
    cells = [(c or "").strip().lower() for c in row]
    return set(cells) >= HEADER_CELLS


def normalize_row(row):
    r = [(c or "").strip() for c in row] + ["", "", ""]
    return r[:3]  # Parameter, Description, Guide


def extract_weight(text: str):
    m = re.search(r"(\d+)\s*%", text or "")
    return int(m.group(1)) if m else None


def make_point_id(job_key: str, term: str, idx: int) -> str:
    name = f"{job_key}::{idx}::{term}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


async def upsert_catalog(jd_pdf_path: str) -> dict:
    """Use LLM to create catalog metadata from JD text. Embed title + each alias separately."""
    ensure_collection(COLLECTION_CATALOG, vector_size=VECTOR_SIZE)
    jd_first_pages = read_pdf_text(jd_pdf_path, max_pages=2)
    raw = f"FILE: {os.path.basename(jd_pdf_path)}\n\n{jd_first_pages[:2000]}"
    meta = await generate_job_catalog_metadata(raw)

    # Collect all searchable terms: title + all aliases
    searchable_terms = [meta['title']] + meta['aliases']

    # Embed all terms in one batch (efficient)
    vectors = await embed_texts_openai(searchable_terms)

    # Create one point per searchable term
    points = []
    for idx, (term, vec) in enumerate(zip(searchable_terms, vectors)):
        point_id = make_point_id(meta["job_key"], term, idx)

        payload = {
            "text": term,
            "doc_type": "job_catalog",
            "job_key": meta["job_key"],
            "title": meta["title"],
            "searchable_term": term,
            "is_primary": (idx == 0),
            "aliases": meta["aliases"],
            "tags": meta["tags"],
            "version": meta.get("version", "v1"),
            "source": os.path.basename(jd_pdf_path),
        }

        points.append({
            "id": point_id,
            "vector": vec,
            "payload": payload
        })

    upsert_points_batch(COLLECTION_CATALOG, points)

    log.info(
        f"Catalog upserted: {meta['title']} → job_key={meta['job_key']} "
        f"({len(points)} searchable terms)"
    )
    return meta


async def ingest_jd_chunks(job_key: str, jd_pdf_path: str):
    ensure_collection(COLLECTION_CV, vector_size=VECTOR_SIZE)
    raw = read_pdf_text(jd_pdf_path)
    chunks = chunk_text(raw, size=1000, overlap=150)
    vecs = await embed_texts_openai(chunks)
    payloads = [{
        "text": t,
        "doc_type": "jd_chunk",
        "job_key": job_key,
        "source": os.path.basename(jd_pdf_path),
        "chunk_index": i
    } for i, t in enumerate(chunks)]
    upsert_texts_with_ids(COLLECTION_CV, vecs, payloads)
    log.info(f"Ingested {len(chunks)} JD chunks for job_key={job_key}")


async def ingest_case_brief(job_key: str, brief_pdf_path: str):
    ensure_collection(COLLECTION_PROJECT, vector_size=VECTOR_SIZE)
    raw = read_pdf_text(brief_pdf_path)
    chunks = chunk_text(raw, size=1000, overlap=150)
    vecs = await embed_texts_with_openai_safe(chunks)
    payloads = [{
        "text": t,
        "doc_type": "case_brief",
        "job_key": job_key,
        "source": os.path.basename(brief_pdf_path),
        "chunk_index": i
    } for i, t in enumerate(chunks)]
    upsert_texts_with_ids(COLLECTION_PROJECT, vecs, payloads)
    log.info(f"Ingested {len(chunks)} case-brief chunks for job_key={job_key}")


async def ingest_rubric(job_key: str, rubric_pdf_path: str):
    ensure_collection(COLLECTION_PROJECT, vector_size=VECTOR_SIZE)
    rows = []
    with pdfplumber.open(rubric_pdf_path) as pdf:
        for page in pdf.pages:
            for tbl in (page.extract_tables() or []):
                rows += [[(c or "").strip() for c in r] for r in tbl if r]

    def is_header(cells: List[str]) -> bool:
        s = set([c.lower() for c in cells])
        return {"parameter", "description", "scoring guide"} <= s

    clean = [r[:3] + ["", "", ""] for r in rows if any(r) and not is_header(r)]
    md = ["# Scoring Rubric (Consolidated)\n"]
    for r in clean:
        param = re.sub(r"\(.*?weight.*?\)", "", r[0], flags=re.I).strip()
        if not param:
            continue
        desc, guide = r[1], r[2]
        md += [f"### {param}", f"**Description:** {desc}" if desc else "",
               f"**Guide:** {guide}" if guide else "", ""]
    consolidated = "\n".join([x for x in md if x is not None])

    blocks = chunk_text(consolidated, size=1800, overlap=200)
    vecs = await embed_texts_with_openai_safe(blocks)
    payloads = [{
        "text": blk,
        "doc_type": "rubric",
        "job_key": job_key,
        "source": os.path.basename(rubric_pdf_path),
        "chunk_index": i,
        "format": "markdown"
    } for i, blk in enumerate(blocks)]
    upsert_texts_with_ids(COLLECTION_PROJECT, vecs, payloads)
    log.info(f"Ingested {len(blocks)} rubric blocks for job_key={job_key}")


async def embed_texts_with_openai_safe(texts: List[str]):
    if not texts:
        return []
    return await embed_texts_openai(texts)


#  main orchestrator
async def main(jd_pdf: str, brief_pdf: str, rubric_pdf: str):
    for p in (jd_pdf, brief_pdf, rubric_pdf):
        if not (os.path.isfile(p) and p.lower().endswith(".pdf")):
            raise FileNotFoundError(f"Missing/invalid PDF: {p}")

    meta = await upsert_catalog(jd_pdf_path=jd_pdf)

    job_key = meta["job_key"]
    log.info(f"Using job_key={job_key}")

    await ingest_jd_chunks(job_key, jd_pdf)
    await ingest_case_brief(job_key, brief_pdf)
    await ingest_rubric(job_key, rubric_pdf)

    log.info(" Ingestion completed successfully.")
    log.info(
        f"Catalog: {meta['title']} ({job_key}) | aliases={meta['aliases']} | tags={meta['tags']}")


async def generate_job_catalog_metadata_from_pdf(jd_pdf: str) -> dict:
    try:
        ensure_collection(COLLECTION_CATALOG, vector_size=VECTOR_SIZE)
        first_pages = read_pdf_text(jd_pdf, max_pages=2)[:2000]
        raw = f"{os.path.basename(jd_pdf)}\n\n{first_pages}"
        meta = await generate_job_catalog_metadata(raw)
        return meta
    except Exception as e:
        raise RuntimeError(f"LLM catalog generation failed: {e}") from e

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Ingest JD + Case Brief + Rubric with unified job_key")
    parser.add_argument("--jd", required=True,
                        help="Path to Job Description PDF")
    parser.add_argument("--brief", required=True,
                        help="Path to Case Study Brief PDF")
    parser.add_argument("--rubric", required=True,
                        help="Path to Scoring Rubric PDF")
    args = parser.parse_args()
    asyncio.run(main(args.jd, args.brief, args.rubric))
