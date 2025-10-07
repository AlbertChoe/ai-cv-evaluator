import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
from app.settings import settings
from domain.schemas import UploadResponse
from infra.repositories.files_repository import FilesRepository

router = APIRouter()
files_repo = FilesRepository()


@router.post("/upload", response_model=UploadResponse)
async def upload(cv: Optional[UploadFile] = File(default=None),
                 report: Optional[UploadFile] = File(default=None)) -> UploadResponse:
    if not cv and not report:
        raise HTTPException(
            status_code=400, detail="Upload at least one file: 'cv' or 'report'")
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    resp = UploadResponse()

    async def save_one(f: UploadFile, ftype: str) -> str:
        name = f.filename or "uploaded.pdf"
        path = os.path.join(settings.STORAGE_DIR, name.replace(" ", "_"))
        content = await f.read()
        with open(path, "wb") as out:
            out.write(content)
        return files_repo.save(ftype=ftype, path=path, name=name)

    if cv:
        resp.cv_id = await save_one(cv, "cv")
    if report:
        resp.report_id = await save_one(report, "report")
    return resp
