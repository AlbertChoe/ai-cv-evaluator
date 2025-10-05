import os
import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")  # <- make sure .env is at root

url = "https://api.openai.com/v1/embeddings"
headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
payload = {"model": "text-embedding-3-small", "input": "hello world"}
r = httpx.post(url, headers=headers, json=payload)
print(r.status_code, r.text)
