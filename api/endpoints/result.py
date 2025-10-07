from fastapi import APIRouter, HTTPException
from domain.schemas import JobStatusResponse
from infra.repositories.jobs_repository import JobsRepository

router = APIRouter()
jobs_repo = JobsRepository()


@router.get("/result/{job_id}", response_model=JobStatusResponse)
async def get_result(job_id: str) -> JobStatusResponse:
    job = jobs_repo.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse(id=job["id"], status=job["status"], result=job.get("result"), error=job.get("error"))
