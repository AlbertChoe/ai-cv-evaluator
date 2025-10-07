import json
import httpx
from typing import Dict, List
from app.settings import settings
from infra.llm.prompts import CATALOG_PROMPT, CV_EVAL_PROMPT, PROJECT_EVAL_PROMPT, FINAL_SUMMARY_PROMPT


async def _openai_chat(messages, model: str):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {settings.OPENAI_API_KEY}"}
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"]


async def _openrouter_chat(messages, model: str):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost",
        "X-Title": settings.APP_NAME,
    }
    payload = {"model": model, "messages": messages, "temperature": 0.2}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"]


async def _choose_and_call(messages):
    if settings.OPENAI_API_KEY:
        return await _openai_chat(messages, settings.OPENAI_MODEL)
    # if settings.OPENROUTER_API_KEY:
    #     return await _openrouter_chat(messages, settings.OPENROUTER_MODEL)
    return json.dumps({"stub": True})


def _parse_json_or_stub(text: str, stub: Dict) -> Dict:
    try:
        return json.loads(text)
    except Exception:
        return stub


async def evaluate_cv_llm(cv_text: str, refs: List[str]) -> Dict:
    content = f"{CV_EVAL_PROMPT}\n\nCV:\n{cv_text[:5000]}\n\nReferences:\n" + "\n---\n".join(
        refs[:5])
    messages = [{"role": "system", "content": "You are a strict evaluator returning only valid JSON."},
                {"role": "user", "content": content}]
    resp = await _choose_and_call(messages)
    return _parse_json_or_stub(resp, {"cv_match_rate": 0.5, "cv_feedback": "Stub feedback."})


async def evaluate_project_llm(report_text: str, refs: List[str]) -> Dict:
    content = f"{PROJECT_EVAL_PROMPT}\n\nReport:\n{report_text[:5000]}\n\nReferences:\n" + "\n---\n".join(
        refs[:5])
    messages = [{"role": "system", "content": "You are a strict evaluator returning only valid JSON."},
                {"role": "user", "content": content}]
    resp = await _choose_and_call(messages)
    return _parse_json_or_stub(resp, {"project_score": 2.5, "project_feedback": "Stub feedback."})


async def summarize_overall_llm(cv_eval: Dict, project_eval: Dict) -> Dict:
    content = f"{FINAL_SUMMARY_PROMPT}\n\nCV Eval JSON: {json.dumps(cv_eval)}\nProject Eval JSON: {json.dumps(project_eval)}"
    messages = [{"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": content}]
    resp = await _choose_and_call(messages)
    return _parse_json_or_stub(resp, {"overall_summary": "Stub overall summary."})


async def generate_job_catalog_metadata(raw_text: str, timeout=30) -> dict:
    payload = {
        "model": settings.OPENAI_MODEL or "gpt-4o-mini",
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You return strict JSON only."},
            {"role": "user", "content": CATALOG_PROMPT.format(raw=raw_text)},
        ],
    }
    async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            timeout=timeout) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        meta = json.loads(content)
        for k in ("title", "aliases", "tags", "job_key"):
            if k not in meta:
                raise ValueError(f"Missing key in catalog JSON: {k}")
        if not isinstance(meta["aliases"], list) or not isinstance(meta["tags"], list):
            raise ValueError("aliases/tags must be arrays")
        return meta
