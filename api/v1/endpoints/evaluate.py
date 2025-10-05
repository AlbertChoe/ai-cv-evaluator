import asyncio
from fastapi import APIRouter, HTTPException
from domain.schemas import EvaluateRequest, JobStatusResponse
from infra.repositories.files_repository import FilesRepository
from infra.repositories.jobs_repository import JobsRepository
from domain.services.evaluation_pipeline import run_evaluation

router = APIRouter()
files_repo = FilesRepository()
jobs_repo = JobsRepository()


@router.post("/evaluate", response_model=JobStatusResponse)
async def evaluate(body: EvaluateRequest) -> JobStatusResponse:
    if not (files_repo.exists(body.cv_id) and files_repo.exists(body.report_id)):
        raise HTTPException(
            status_code=404, detail="cv_id or report_id not found")

    job_id = jobs_repo.create_job(body.job_title, body.cv_id, body.report_id)

    async def runner():
        try:
            jobs_repo.update_status(job_id, "processing")
            cv_path = files_repo.get_path(body.cv_id)
            report_path = files_repo.get_path(body.report_id)
            result = await run_evaluation(body.job_title, cv_path, report_path)
            jobs_repo.complete(job_id, result)
        except Exception as e:
            jobs_repo.fail(job_id, str(e))

    asyncio.create_task(runner())
    return JobStatusResponse(id=job_id, status="queued")
