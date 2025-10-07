import asyncio
import os
import re
import json
import pdfplumber
from infra.rag.qdrant_client import ensure_collection, COLLECTION_PROJECT, upsert_texts_with_ids
from infra.rag.embeddings import embed_texts_openai

JOB_KEY = "backend_pe_v1"
PDF_PATH = "data/scoring_rubric_backend.pdf"

HEADER_CELLS = {"parameter", "description", "scoring guide"}


def is_header_row(row):
    cells = [(c or "").strip().lower() for c in row]
    return set(cells) >= HEADER_CELLS


def normalize_row(row):
    r = [(c or "").strip() for c in row] + ["", "", ""]
    return r[:3]  # Parameter, Description, Guide


def extract_weight(text: str):
    m = re.search(r"(\d+)\s*%", text or "")
    return int(m.group(1)) if m else None


def soft_chunk(text: str, max_chars=1800, overlap=200):
    out, i = [], 0
    while i < len(text):
        piece = text[i:i+max_chars].strip()
        if piece:
            out.append(piece)
        i += max(1, max_chars - overlap)
    return out


async def main():
    ensure_collection(COLLECTION_PROJECT, vector_size=1536)

    rows = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            for tbl in (page.extract_tables() or []):
                for r in tbl:
                    if not r:
                        continue
                    rows.append(normalize_row(r))

    clean_rows = [r for r in rows if any(r) and not is_header_row(r)]

    md_lines = ["# Scoring Rubric (Consolidated)\n"]
    json_rows = []
    for r in clean_rows:
        param_raw = r[0]
        if not param_raw:
            continue
        weight = extract_weight(param_raw)
        param = re.sub(r"\(.*?weight.*?\)", "", param_raw, flags=re.I).strip()
        desc, guide = r[1], r[2]

        if param.lower().startswith(("scoring rubric", "cv match evaluation", "project deliverable evaluation", "overall candidate evaluation")):
            # treat as section label
            md_lines += [f"## {param}", ""]
            continue

        md_lines += [
            f"### {param}" +
            (f" (Weight: {weight}%)" if weight is not None else ""),
            (f"**Description:** {desc}" if desc else ""),
            (f"**Guide:** {guide}" if guide else ""),
            ""
        ]
        json_rows.append({
            "param_name": param, "weight_pct": weight,
            "description": desc, "guide": guide
        })

    consolidated_markdown = "\n".join(
        [ln for ln in md_lines if ln is not None])

    # Full chunk first, then split only if long
    MAX_CHARS = 1800
    OVERLAP = 200
    blocks = soft_chunk(consolidated_markdown,
                        max_chars=MAX_CHARS, overlap=OVERLAP)

    chunks = blocks
    payloads = [{
        "text": blk,
        "doc_type": "rubric",
        "job_key": JOB_KEY,
        "source": os.path.basename(PDF_PATH),
        "chunk_index": i,
        "row_count": len(json_rows),
        "format": "markdown",
    } for i, blk in enumerate(blocks)]

    # Preview
    # print(
    #     f"\n=== CONSOLIDATED PREVIEW: {len(blocks)} block(s), rows={len(json_rows)} ===\n")
    # for i, p in enumerate(payloads):
    #     print(f"--- Block #{i} ---")
    #     print(p["text"])
    #     print()

    vecs = await embed_texts_openai(chunks)
    upsert_texts_with_ids(COLLECTION_PROJECT, vecs, payloads)
    print(
        f"Ingested {len(chunks)} consolidated rubric block(s) for job_key={JOB_KEY}")

if __name__ == "__main__":
    asyncio.run(main())
