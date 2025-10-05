import os
from pydantic import BaseModel
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseModel):
    APP_NAME: str = os.getenv("APP_NAME", "AI CV & Project Evaluator")
    ENV: str = os.getenv("ENV", "development")
    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "storage")
    API_V1_PREFIX: str = os.getenv("API_V1_PREFIX", "/api/v1")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    SQLITE_PATH: str = os.getenv("SQLITE_PATH", "app.sqlite3")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY") or None
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY") or None
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()