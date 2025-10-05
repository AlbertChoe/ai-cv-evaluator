from typing import List
import httpx
from app.settings import settings


async def embed_texts_openai(texts: List[str]) -> List[List[float]]:
    api_key = settings.OPENAI_API_KEY
    model = settings.OPENAI_EMBEDDING_MODEL
    if not api_key:
        return [[0.0]*8 for _ in texts]
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "input": texts}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return [item["embedding"] for item in data["data"]]
