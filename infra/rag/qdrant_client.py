from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.settings import settings
import uuid

COLLECTION_CV = "job_descriptions"
COLLECTION_PROJECT = "case_and_rubrics"


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)


def ensure_collection(name: str, vector_size: int = 1536):
    c = get_client()
    if name not in [x.name for x in c.get_collections().collections]:
        c.create_collection(collection_name=name, vectors_config=VectorParams(
            size=vector_size, distance=Distance.COSINE))


def upsert_texts(collection: str, vectors: list[list[float]], payloads: list[dict]):
    c = get_client()
    points = [PointStruct(id=str(uuid.uuid4()), vector=v, payload=p)
              for v, p in zip(vectors, payloads)]
    c.upsert(collection_name=collection, points=points)


def search_top_k(collection: str, query_vector: list[float], k: int = 5) -> list[dict]:
    c = get_client()
    res = c.search(collection_name=collection,
                   query_vector=query_vector, limit=k)
    return [r.payload for r in res]
