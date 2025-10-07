import asyncio
import json
from typing import Dict, List, Type, TypeVar

import httpx
from pydantic import BaseModel, Field, ValidationError, validator

from app.settings import settings
from infra.llm.prompts import (
    CATALOG_PROMPT,
    CV_EVAL_PROMPT,
    FINAL_SUMMARY_PROMPT,
    PROJECT_EVAL_PROMPT,
)

T = TypeVar("T", bound=BaseModel)


class CVEvaluationPayload(BaseModel):
    cv_match_rate: float = Field(..., ge=0.0, le=1.0)
    cv_feedback: List[str]

    @validator("cv_feedback", pre=True)
    def _ensure_list(cls, value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        raise ValueError("cv_feedback must be a string or list of strings")


class ProjectEvaluationPayload(BaseModel):
    project_score: float = Field(..., ge=1.0, le=5.0)
    project_feedback: List[str]

    @validator("project_feedback", pre=True)
    def _ensure_list(cls, value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]
        raise ValueError(
            "project_feedback must be a string or list of strings")


class SummaryPayload(BaseModel):
    overall_summary: str = Field(..., min_length=1)


async def _post_with_retries(
    url: str,
    headers: Dict[str, str],
    payload: Dict,
    *,
    timeout: int = 15,
    max_attempts: int = 3,
) -> Dict:
    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            retriable = status >= 500 or status in {408, 429}
            if not retriable or attempt == max_attempts:
                raise
        except httpx.RequestError:
            if attempt == max_attempts:
                raise
        await asyncio.sleep(backoff)
        backoff *= 2
    raise RuntimeError("Unexpected retry exhaustion")


async def _openai_chat(messages, model: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    data = await _post_with_retries(url, headers, payload)
    return data["choices"][0]["message"]["content"]


async def _openrouter_chat(messages, model: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost",
        "X-Title": settings.APP_NAME,
    }
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    data = await _post_with_retries(url, headers, payload)
    return data["choices"][0]["message"]["content"]


async def _choose_and_call(messages) -> str:
    if settings.OPENAI_API_KEY:
        return await _openai_chat(messages, settings.OPENAI_MODEL)
    if settings.OPENROUTER_API_KEY:
        return await _openrouter_chat(messages, settings.OPENROUTER_MODEL)
    raise RuntimeError("No LLM provider configured")


def _validate_llm_response(raw_text: str, model: Type[T]) -> T:
    try:
        return model.parse_raw(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response was not valid JSON") from exc
    except ValidationError as exc:
        raise ValueError(f"LLM response failed validation: {exc}") from exc


async def evaluate_cv_llm(cv_text: str, refs: List[str]) -> Dict:
    content = f"{CV_EVAL_PROMPT}\n\nCV:\n{cv_text[:5000]}\n\nReferences:\n" + "\n---\n".join(
        refs[:5]
    )
    messages = [
        {"role": "system", "content": "You are a strict evaluator returning only valid JSON."},
        {"role": "user", "content": content},
    ]
    try:
        resp = await _choose_and_call(messages)
    except RuntimeError:
        return CVEvaluationPayload(cv_match_rate=0.5, cv_feedback=["Stub feedback."]).dict()
    parsed = _validate_llm_response(resp, CVEvaluationPayload)
    return parsed.dict()


async def evaluate_project_llm(report_text: str, refs: List[str]) -> Dict:
    content = f"{PROJECT_EVAL_PROMPT}\n\nReport:\n{report_text[:5000]}\n\nReferences:\n" + "\n---\n".join(
        refs[:5]
    )
    messages = [
        {"role": "system", "content": "You are a strict evaluator returning only valid JSON."},
        {"role": "user", "content": content},
    ]
    try:
        resp = await _choose_and_call(messages)
    except RuntimeError:
        return ProjectEvaluationPayload(
            project_score=2.5, project_feedback=["Stub feedback."]
        ).dict()
    parsed = _validate_llm_response(resp, ProjectEvaluationPayload)
    return parsed.dict()


async def summarize_overall_llm(cv_eval: Dict, project_eval: Dict) -> Dict:
    content = (
        f"{FINAL_SUMMARY_PROMPT}\n\nCV Eval JSON: {json.dumps(cv_eval)}\nProject Eval JSON: {json.dumps(project_eval)}"
    )
    messages = [
        {"role": "system", "content": "Return only valid JSON."},
        {"role": "user", "content": content},
    ]
    try:
        resp = await _choose_and_call(messages)
    except RuntimeError:
        return SummaryPayload(overall_summary="Stub overall summary.").dict()
    parsed = _validate_llm_response(resp, SummaryPayload)
    return parsed.dict()


async def generate_job_catalog_metadata(raw_text: str, timeout=30) -> dict:
    payload = {
        "model": settings.OPENAI_MODEL,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You return strict JSON only."},
            {"role": "user", "content": CATALOG_PROMPT.format(raw=raw_text)},
        ],
    }
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
    data = await _post_with_retries(
        "https://api.openai.com/v1/chat/completions",
        headers,
        payload,
        timeout=timeout,
    )
    content = data["choices"][0]["message"]["content"]
    meta = json.loads(content)
    for k in ("title", "aliases", "tags", "job_key"):
        if k not in meta:
            raise ValueError(f"Missing key in catalog JSON: {k}")
    if not isinstance(meta["aliases"], list) or not isinstance(meta["tags"], list):
        raise ValueError("aliases/tags must be arrays")
    return meta
