import asyncio
from typing import List
from infra.rag.embeddings import embed_texts_openai
from infra.rag.qdrant_client import COLLECTION_CV, COLLECTION_PROJECT, search_top_k

async def retrieve_for_cv(job_title: str) -> List[str]:
    qvecs = await embed_texts_openai([f"job requirements for {job_title}"])
    payloads = search_top_k(COLLECTION_CV, qvecs[0], k=5)
    return [p.get("text", "") for p in payloads]

async def retrieve_for_project() -> List[str]:
    qvecs = await embed_texts_openai(["case study brief and project scoring rubric"])
    payloads = search_top_k(COLLECTION_PROJECT, qvecs[0], k=5)
    return [p.get("text", "") for p in payloads]

def retrieve_for_cv_sync(job_title: str) -> List[str]:
    return asyncio.run(retrieve_for_cv(job_title))

def retrieve_for_project_sync() -> List[str]:
    return asyncio.run(retrieve_for_project())