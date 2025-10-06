import asyncio
from typing import List, Dict
from infra.rag.embeddings import embed_texts_openai
from infra.rag.qdrant_client import COLLECTION_CV, COLLECTION_PROJECT, search_top_k_filtered, fetch_neighbors_by_index


def _stitch(hits: List[Dict], collection: str, job_key: str, radius: int = 1) -> List[Dict]:
    # For each hit, pull neighbor chunks and merge.
    stitched = []
    seen_keys = set()
    for h in hits:
        p = h["payload"]
        key = (p.get("source"), p.get("doc_type"), p.get("chunk_index"))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        neighbors = fetch_neighbors_by_index(
            collection=collection,
            job_key=job_key,
            doc_type=p.get("doc_type"),
            source=p.get("source"),
            center_index=p.get("chunk_index", 0),
            radius=radius
        )
        # Merge neighbors into one block
        text_block = "\n".join(n.get("text", "")
                               for n in neighbors if n.get("text"))
        stitched.append({
            "text": text_block,
            "source": p.get("source"),
            "doc_type": p.get("doc_type"),
            "start_chunk_index": min(n.get("chunk_index", 0) for n in neighbors) if neighbors else p.get("chunk_index", 0)
        })
    return stitched


async def retrieve_for_cv(job_key: str, job_title: str, k: int = 5, radius: int = 1) -> List[str]:
    q = f"job requirements for {job_title}"
    qvec = (await embed_texts_openai([q]))[0]
    hits = search_top_k_filtered(
        COLLECTION_CV, qvec, k=k, job_key=job_key, doc_types=["jd"])
    blocks = _stitch(hits, COLLECTION_CV, job_key, radius=radius)
    return [b["text"] for b in blocks]


async def retrieve_for_project(job_key: str, k: int = 5, radius: int = 1) -> List[str]:
    q = "case study brief and project scoring rubric for this job"
    qvec = (await embed_texts_openai([q]))[0]
    hits = search_top_k_filtered(
        COLLECTION_PROJECT, qvec, k=k, job_key=job_key, doc_types=["case_brief", "rubric"])
    blocks = _stitch(hits, COLLECTION_PROJECT, job_key, radius=radius)
    return [b["text"] for b in blocks]


def retrieve_for_cv_sync(job_key: str, job_title: str, k: int = 5) -> List[str]:
    return asyncio.run(retrieve_for_cv(job_key, job_title, k))


def retrieve_for_project_sync(job_key: str, k: int = 5) -> List[str]:
    return asyncio.run(retrieve_for_project(job_key, k))
