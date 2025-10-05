import asyncio
import os
import pdfplumber
from infra.rag.qdrant_client import ensure_collection, COLLECTION_CV, upsert_texts
from infra.rag.embeddings import embed_texts_openai


def chunk_text(text: str, size=1000, overlap=150):
    chunks = []
    i = 0
    while i < len(text):
        piece = text[i:i+size]
        piece = piece.strip()
        if piece:
            chunks.append(piece)
        i += max(1, size - overlap)
    return chunks


PDF_PATH = "data/job_description_backend.pdf"


def parse_pdf_text(path: str) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            parts.append(p.extract_text() or "")
    return "\n".join(parts)


async def main():
    ensure_collection(COLLECTION_CV, vector_size=1536)
    raw = parse_pdf_text(PDF_PATH)
    chunks = chunk_text(raw, size=1000, overlap=150)
    vecs = await embed_texts_openai(chunks)
    payloads = [{
        "text": t,
        "doc_type": "jd",
        "source": os.path.basename(PDF_PATH),
        "job_title": "Product Engineer (Backend)"
    } for t in chunks]
    upsert_texts(COLLECTION_CV, vecs, payloads)
    print(f"Ingested {len(chunks)} JD chunks from {PDF_PATH}.")

if __name__ == "__main__":
    asyncio.run(main())
