import json
import logging
from typing import Dict, List
from infra.pdf.parser import parse_pdf_text
from infra.rag.retriever import retrieve_for_cv_sync, retrieve_for_project_sync, retrieve_for_cv, retrieve_for_project
from infra.llm.client import evaluate_cv_llm, evaluate_project_llm, summarize_overall_llm

logger = logging.getLogger("evaluation_pipeline")
logger.setLevel(logging.INFO)

# write logs to file for manual inspection
fh = logging.FileHandler("evaluation_debug.log", mode="a", encoding="utf-8")
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)


async def run_evaluation(job_title: str, cv_path: str, report_path: str) -> Dict:
    logger.info("=== Starting evaluation job ===")
    logger.info(f"Job title: {job_title}")
    logger.info(f"CV path: {cv_path}")
    logger.info(f"Report path: {report_path}")

    # Parse PDFs
    cv_text = parse_pdf_text(cv_path)
    report_text = parse_pdf_text(report_path)
    logger.info(f"CV text length: {len(cv_text)} chars")
    logger.info(f"Report text length: {len(report_text)} chars")

    # reference texts from Qdrant
    logger.info("Retrieving job description references for CV")
    cv_refs: List[str] = await retrieve_for_cv(job_title=job_title)
    logger.info(f"Retrieved {len(cv_refs)} CV references")
    for i, ref in enumerate(cv_refs[:3]):
        logger.info(f"CV ref {i+1}: {ref}")

    logger.info("Retrieving case brief + rubric references for project")
    proj_refs: List[str] = await retrieve_for_project()
    logger.info(f"Retrieved {len(proj_refs)} project references")
    for i, ref in enumerate(proj_refs[:3]):
        logger.info(f"Project ref {i+1}: {ref}")

    # Evaluate CV
    logger.info("Calling LLM for CV evaluation")
    cv_eval = await evaluate_cv_llm(cv_text=cv_text, refs=cv_refs)
    logger.info(f"CV evaluation result:\n{json.dumps(cv_eval, indent=2)}")

    # Evaluate Project Report
    logger.info("Calling LLM for Project evaluation")
    project_eval = await evaluate_project_llm(report_text=report_text, refs=proj_refs)
    logger.info(
        f"Project evaluation result:\n{json.dumps(project_eval, indent=2)}")

    # Summarize
    logger.info("Calling LLM for overall summary synthesis")
    summary = await summarize_overall_llm(cv_eval=cv_eval, project_eval=project_eval)
    logger.info(f"Final summary:\n{json.dumps(summary, indent=2)}")

    #  Aggregate result
    result = {
        "cv_match_rate": cv_eval.get("cv_match_rate", 0.0),
        "cv_feedback": cv_eval.get("cv_feedback", ""),
        "project_score": project_eval.get("project_score", 0.0),
        "project_feedback": project_eval.get("project_feedback", ""),
        "overall_summary": summary.get("overall_summary", ""),
    }

    logger.info(f"Final combined result:\n{json.dumps(result, indent=2)}")
    logger.info("=== Evaluation job completed ===\n")
    return result
