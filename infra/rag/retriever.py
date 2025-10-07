import logging
from typing import Optional, Tuple, List, Dict
import asyncio
from infra.rag.embeddings import embed_texts_openai
from infra.rag.qdrant_client import COLLECTION_CATALOG, COLLECTION_CV, COLLECTION_PROJECT, search_top_k_filtered, fetch_neighbors_by_index

logger = logging.getLogger("evaluation_pipeline")
logger.setLevel(logging.INFO)


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


async def retrieve_rubrics(
    job_key: str,
    k: int = 5,
    radius: int = 1,
    qvec: Optional[List[float]] = None,
) -> List[str]:
    if qvec is None:
        qvec = (await embed_texts_openai(["scoring rubric for evaluation"]))[0]
    rb_hits = search_top_k_filtered(
        COLLECTION_PROJECT,
        qvec,
        k=k,
        job_key=job_key,
        doc_types=["rubric"],
    )
    rb_blocks = _stitch(rb_hits, COLLECTION_PROJECT, job_key, radius=radius)
    return [b["text"] for b in rb_blocks]


async def retrieve_for_cv(
    job_key: str,
    job_title: str,
    job_tags: Optional[List[str]] = None,
    k: int = 5,
    radius: int = 1,
    rubric_blocks: Optional[List[str]] = None,
    qvec: Optional[List[float]] = None,
) -> List[str]:
    tag_str = f" relevant tags: {', '.join(job_tags)}" if job_tags else ""
    if qvec is None:
        qvec = (await embed_texts_openai([f"job requirements and evaluation criteria for {job_title}{tag_str}"]))[0]

    jd_hits = search_top_k_filtered(
        COLLECTION_CV, qvec, k=k, job_key=job_key, doc_types=["jd_chunk"]
    )
    jd_blocks = _stitch(jd_hits, COLLECTION_CV, job_key, radius=radius)

    if rubric_blocks is None:
        rubric_blocks = await retrieve_rubrics(job_key=job_key, k=k, radius=radius)

    blocks = [b["text"] for b in jd_blocks] + rubric_blocks
    return blocks


async def retrieve_for_project(
    job_key: str,
    job_title: Optional[str] = None,
    job_tags: Optional[List[str]] = None,
    k: int = 5,
    radius: int = 1,
    rubric_blocks: Optional[List[str]] = None,
    qvec: Optional[List[float]] = None,
) -> List[str]:
    tag_str = f" relevant tags: {', '.join(job_tags)}" if job_tags else ""
    role_str = f" for {job_title}" if job_title else ""
    if qvec is None:
        qvec = (await embed_texts_openai([f"case study brief and project scoring rubric{role_str}{tag_str}"]))[0]

    brief_hits = search_top_k_filtered(
        COLLECTION_PROJECT, qvec, k=k, job_key=job_key, doc_types=[
            "case_brief"]
    )
    brief_blocks = _stitch(
        brief_hits, COLLECTION_PROJECT, job_key, radius=radius)

    if rubric_blocks is None:
        rubric_blocks = await retrieve_rubrics(job_key=job_key, k=k, radius=radius)

    blocks = rubric_blocks + [b["text"] for b in brief_blocks]
    return blocks


def retrieve_for_cv_sync(job_key: str, job_title: str, k: int = 5) -> List[str]:
    return asyncio.run(retrieve_for_cv(job_key, job_title, k))


def retrieve_for_project_sync(job_key: str, k: int = 5) -> List[str]:
    return asyncio.run(retrieve_for_project(job_key, k))


async def resolve_job_key(
    job_title: str,
    min_similarity: float = 0.80,
) -> Tuple[Optional[str], float, List[Dict]]:
    """Resolve job title to job_key using semantic search on individual terms."""
    [qvec] = await embed_texts_openai([job_title])

    hits = search_top_k_filtered(
        collection=COLLECTION_CATALOG,
        query_vector=qvec,
        k=5,
        job_key=None,
        doc_types=["job_catalog"],
    )

    candidates: List[Dict] = []
    seen_job_keys = set()

    for h in hits:
        p = h.get("payload", {})
        jk = p.get("job_key")

        # Deduplicate: same job_key may appear multiple times (different aliases)
        if jk in seen_job_keys:
            continue
        seen_job_keys.add(jk)

        similarity = float(h.get("score", 0.0))
        candidates.append({
            "job_key": jk,
            "title": p.get("title"),
            "matched_term": p.get("searchable_term"),
            "is_primary": p.get("is_primary", False),
            "tags": p.get("tags", []),
            "similarity": similarity,
        })

    if not candidates:
        return None, 0.0, []

    top = max(candidates, key=lambda x: x["similarity"])

    # Accept if similarity is high (should be 0.90+ for exact alias matches)
    if top["job_key"] and top["similarity"] >= min_similarity:
        logger.info(
            f"✓ Matched '{job_title}' → {top['job_key']} "
            f"(term: '{top['matched_term']}', similarity: {top['similarity']:.3f})"
        )
        return top["job_key"], top["similarity"], candidates

    # logger near-miss for debugging
    logger.warning(
        f"✗ No confident match for '{job_title}' "
        f"(best: {top['matched_term']} @ {top['similarity']:.3f})"
    )
    return None, top["similarity"], candidates
