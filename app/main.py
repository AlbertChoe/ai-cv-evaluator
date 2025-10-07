from fastapi import FastAPI
from app.settings import settings
from app.logging import configure_logging
from app.error_handlers import attach_error_handlers
from api.router import api_router
from infra.db.session import init_db

configure_logging()
app = FastAPI(title=settings.APP_NAME)


@app.on_event("startup")
def _on_startup():
    init_db()


attach_error_handlers(app)
app.include_router(api_router)
