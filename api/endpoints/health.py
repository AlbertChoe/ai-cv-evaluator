from fastapi import APIRouter, HTTPException
from infra.rag.qdrant_client import get_client

router = APIRouter()


@router.get("/vector-db/health")
def vector_db_health():
    client = get_client()
    try:
        collections = client.get_collections()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {
        "status": "ok",
        "collections": [col.name for col in collections.collections],
        "collection_count": len(collections.collections),
    }
