from typing import Iterable, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, Range
from app.settings import settings
import hashlib

COLLECTION_CV = "job_descriptions"
COLLECTION_PROJECT = "case_and_rubrics"


def get_client():
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)


def _ensure_payload_indexes(collection: str):
    c = get_client()
    for field, schema in [
        ("job_key", "keyword"),
        ("doc_type", "keyword"),
        ("source", "keyword"),
        ("chunk_index", "integer"),
    ]:
        try:
            c.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=schema
            )
        except Exception:
            pass


def ensure_collection(name: str, vector_size: int = 1536):
    c = get_client()
    names = {x.name for x in c.get_collections().collections}
    if name not in names:
        c.create_collection(collection_name=name, vectors_config=VectorParams(
            size=vector_size, distance=Distance.COSINE))
    _ensure_payload_indexes(name)


def _stable_id(job_key: str, doc_type: str, text: str, source: str = "", chunk_index: int = -1) -> str:
    raw = f"{job_key}|{doc_type}|{source}|{chunk_index}|{text}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def upsert_texts_with_ids(collection: str, vectors: list[list[float]], payloads: list[dict]):
    points = [
        PointStruct(
            id=_stable_id(
                p["job_key"], p["doc_type"], p["text"], p.get(
                    "source", ""), p.get("chunk_index", -1)
            ),
            vector=v,
            payload=p
        )
        for v, p in zip(vectors, payloads)
    ]
    get_client().upsert(collection_name=collection, points=points)


def search_top_k_filtered(
    collection: str,
    query_vector: list[float],
    k: int,
    job_key: str,
    doc_types: Optional[Iterable[str]] = None,
):
    must = [FieldCondition(key="job_key", match=MatchValue(value=job_key))]
    if doc_types:
        must.append(FieldCondition(key="doc_type",
                    match=MatchAny(any=list(doc_types))))
    q_filter = Filter(must=must)

    hits = get_client().search(
        collection_name=collection,
        query_vector=query_vector,
        limit=k,
        query_filter=q_filter,
    )
    return [{"payload": h.payload, "score": float(h.score)} for h in hits]


def fetch_neighbors_by_index(
    collection: str,
    job_key: str,
    doc_type: str,
    source: str,
    center_index: int,
    radius: int = 1,
):
    """Return chunks with chunk_index in [center_index - radius, center_index + radius] from same doc."""
    must = [
        FieldCondition(key="job_key", match=MatchValue(value=job_key)),
        FieldCondition(key="doc_type", match=MatchValue(value=doc_type)),
        FieldCondition(key="source", match=MatchValue(value=source)),
        FieldCondition(key="chunk_index", range=Range(
            gte=max(0, center_index - radius),
            lte=center_index + radius
        ))
    ]
    flt = Filter(must=must)
    out = []
    next_page = None
    while True:
        res = get_client().scroll(collection_name=collection,
                                  scroll_filter=flt, limit=256, offset=next_page)
        out.extend([p.payload for p in res[0]])
        next_page = res[1]
        if next_page is None:
            break
    out.sort(key=lambda x: x.get("chunk_index", 0))
    return out
