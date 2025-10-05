from fastapi import APIRouter
from api.v1.endpoints.upload import router as upload_router
from api.v1.endpoints.evaluate import router as evaluate_router
from api.v1.endpoints.result import router as result_router

api_router = APIRouter()
api_router.include_router(upload_router, tags=["upload"])
api_router.include_router(evaluate_router, tags=["evaluate"])
api_router.include_router(result_router, tags=["result"])