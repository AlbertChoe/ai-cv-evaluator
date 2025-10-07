from fastapi import APIRouter
from api.endpoints.upload import router as upload_router
from api.endpoints.evaluate import router as evaluate_router
from api.endpoints.result import router as result_router
from api.endpoints.health import router as health_router

api_router = APIRouter()
api_router.include_router(upload_router, tags=["upload"])
api_router.include_router(evaluate_router, tags=["evaluate"])
api_router.include_router(result_router, tags=["result"])
api_router.include_router(health_router, tags=["health"])
