from pydantic import BaseModel, Field
from typing import Optional, Dict

class UploadResponse(BaseModel):
    cv_id: Optional[str] = None
    report_id: Optional[str] = None

class EvaluateRequest(BaseModel):
    job_title: str = Field(...)
    cv_id: str
    report_id: str

class JobStatusResponse(BaseModel):
    id: str
    status: str
    result: Optional[Dict] = None
    error: Optional[str] = None