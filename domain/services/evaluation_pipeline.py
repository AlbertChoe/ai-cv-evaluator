from typing import Dict, List
from infra.pdf.parser import parse_pdf_text
from infra.rag.retriever import retrieve_for_cv_sync, retrieve_for_project_sync
from infra.llm.client import evaluate_cv_llm, evaluate_project_llm, summarize_overall_llm
import asyncio

async def run_evaluation(job_title: str, cv_path: str, report_path: str) -> Dict:
    cv_text = parse_pdf_text(cv_path)
    report_text = parse_pdf_text(report_path)

    cv_refs: List[str] = retrieve_for_cv_sync(job_title=job_title)
    proj_refs: List[str] = retrieve_for_project_sync()

    cv_eval = await evaluate_cv_llm(cv_text=cv_text, refs=cv_refs)
    project_eval = await evaluate_project_llm(report_text=report_text, refs=proj_refs)
    summary = await summarize_overall_llm(cv_eval=cv_eval, project_eval=project_eval)

    return {
        "cv_match_rate": cv_eval.get("cv_match_rate", 0.0),
        "cv_feedback": cv_eval.get("cv_feedback", ""),
        "project_score": project_eval.get("project_score", 0.0),
        "project_feedback": project_eval.get("project_feedback", ""),
        "overall_summary": summary.get("overall_summary", ""),
    }