from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

def attach_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})