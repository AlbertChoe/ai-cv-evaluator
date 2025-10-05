import uuid
import json
from typing import Optional, Dict, Any
from infra.db.session import SessionLocal
from infra.db.models import JobRecord, JobResultRecord


def _to_text(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, list):
        return "\n".join(f"- {str(x)}" for x in val)
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


class JobsRepository:
    def create_job(self, job_title: str, cv_id: str, report_id: str) -> str:
        jid = f"job_{uuid.uuid4().hex}"
        with SessionLocal() as s:
            s.add(JobRecord(id=jid, status="queued", job_title=job_title,
                            cv_file_id=cv_id, report_file_id=report_id))
            s.commit()
        return jid

    def update_status(self, job_id: str, status: str) -> None:
        with SessionLocal() as s:
            job = s.get(JobRecord, job_id)
            if not job:
                return
            job.status = status
            s.commit()

    def complete(self, job_id: str, result: Dict) -> None:
        with SessionLocal() as s:
            job = s.get(JobRecord, job_id)
            if not job:
                return
            job.status = "completed"
            jr = JobResultRecord(
                job_id=job_id,
                cv_match_rate=float(result.get("cv_match_rate") or 0.0),
                cv_feedback=_to_text(result.get("cv_feedback")),
                project_score=float(result.get("project_score") or 0.0),
                project_feedback=_to_text(result.get("project_feedback")),
                overall_summary=_to_text(result.get("overall_summary")),
            )
            s.add(jr)
            s.commit()

    def fail(self, job_id: str, error: str) -> None:
        with SessionLocal() as s:
            job = s.get(JobRecord, job_id)
            if not job:
                return
            job.status = "failed"
            jr = JobResultRecord(
                job_id=job_id, overall_summary=f"ERROR: {error}")
            s.merge(jr)
            s.commit()

    def get(self, job_id: str) -> Optional[Dict]:
        with SessionLocal() as s:
            job = s.get(JobRecord, job_id)
            if not job:
                return None
            jr = s.get(JobResultRecord, job_id)
            out = {"id": job.id, "status": job.status,
                   "result": None, "error": None}
            if jr and job.status == "completed":
                out["result"] = {
                    "cv_match_rate": jr.cv_match_rate,
                    "cv_feedback": jr.cv_feedback,
                    "project_score": jr.project_score,
                    "project_feedback": jr.project_feedback,
                    "overall_summary": jr.overall_summary,
                }
            if jr and job.status == "failed" and jr.overall_summary:
                out["error"] = jr.overall_summary
            return out
