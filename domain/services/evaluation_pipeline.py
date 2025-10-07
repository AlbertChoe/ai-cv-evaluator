import re
import json
import logging
from typing import Dict, List, Optional

from infra.pdf.parser import parse_pdf_text
from infra.rag.qdrant_client import debug_count, debug_list_collections, debug_scroll_one
from infra.rag.retriever import (
    retrieve_for_cv,
    retrieve_for_project,
    resolve_job_key,
    retrieve_rubrics,
)
from infra.llm.client import (
    evaluate_cv_llm,
    evaluate_project_llm,
    summarize_overall_llm,
)

logger = logging.getLogger("evaluation_pipeline")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("evaluation_debug.log", mode="a", encoding="utf-8")
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)


def redact_numeric_examples(text: str) -> str:
    # remove json-like examples with numeric scores to prevent bias
    text = re.sub(r'\{[^{}]{0,200}("project_score"|\'project_score\')[^{}]+\}',
                  '[redacted-example]', text, flags=re.I | re.S)
    text = re.sub(r'\{[^{}]{0,200}("cv_match_rate"|\'cv_match_rate\')[^{}]+\}',
                  '[redacted-example]', text, flags=re.I | re.S)
    return text


def sanitize_refs(refs: List[str]) -> List[str]:
    return [redact_numeric_examples(r) for r in refs]


async def run_evaluation(job_title: str, cv_path: str, report_path: str) -> Dict:
    logger.info("=== Starting evaluation job ===")
    logger.info(f"Job title: {job_title}")
    logger.info(f"CV path: {cv_path}")
    logger.info(f"Report path: {report_path}")

    job_key, confidence, candidates = await resolve_job_key(job_title)
    job_tags: Optional[List[str]] = None

    if not job_key:
        job_key = job_title.strip().lower().replace(" ", "-")
        logger.warning(
            f"Could not confidently resolve job_key (best similarity={confidence:.3f}). "
            f"Falling back to '{job_key}'. Candidates={candidates}"
        )
    else:
        top = max(candidates, key=lambda x: x["similarity"])
        job_tags = top.get("tags", [])
        logger.info(
            f"Resolved job_key: {job_key} "
            f"(similarity={confidence:.3f}, tags={job_tags})"
        )

    cv_text = parse_pdf_text(cv_path)
    report_text = parse_pdf_text(report_path)
    logger.info(f"CV text length: {len(cv_text)} chars")
    logger.info(f"Report text length: {len(report_text)} chars")

    logger.info("Retrieving shared rubric content")
    rubric_blocks = await retrieve_rubrics(job_key=job_key, k=5, radius=1)
    logger.info(f"Retrieved {len(rubric_blocks)} rubric blocks")

    logger.info("Retrieving job description references for CV")
    cv_refs: List[str] = await retrieve_for_cv(
        job_key=job_key,
        job_title=job_title,
        job_tags=job_tags,
        k=5,
        radius=1,
        rubric_blocks=rubric_blocks,
    )
    cv_refs = sanitize_refs(cv_refs)
    logger.info(f"Retrieved {len(cv_refs)} CV references")
    for i, ref in enumerate(cv_refs[:3]):
        logger.info(f"CV ref {i+1}: {ref[:200] }...")

    logger.info("Retrieving case brief + rubric references for project")
    proj_refs: List[str] = await retrieve_for_project(
        job_key=job_key,
        job_title=job_title,
        job_tags=job_tags,
        k=5,
        radius=1,
        rubric_blocks=rubric_blocks,
    )
    proj_refs = sanitize_refs(proj_refs)
    logger.info(f"Retrieved {len(proj_refs)} project references")
    for i, ref in enumerate(proj_refs[:3]):
        logger.info(f"Project ref {i+1}: {ref[:200] }...")

    #  Evaluate with LLMs
    logger.info("Calling LLM for CV evaluation")
    cv_eval = await evaluate_cv_llm(cv_text=cv_text, refs=cv_refs)
    logger.info(
        "CV evaluation result: "
        f"match_rate={cv_eval.get('cv_match_rate')} feedback_preview={str(cv_eval.get('cv_feedback'))}"
    )

    logger.info("Calling LLM for Project evaluation")
    project_eval = await evaluate_project_llm(report_text=report_text, refs=proj_refs)
    logger.info(
        "Project evaluation result: "
        f"score={project_eval.get('project_score')} feedback_preview={str(project_eval.get('project_feedback'))}"
    )

    #  Summarize
    logger.info("Calling LLM for overall summary synthesis")
    summary = await summarize_overall_llm(cv_eval=cv_eval, project_eval=project_eval)
    logger.info(
        "Overall summary preview: "
        f"{summary.get('overall_summary', '')}"
    )

    result = {
        "cv_match_rate": float(cv_eval.get("cv_match_rate", 0.0) or 0.0),
        "cv_feedback": str(cv_eval.get("cv_feedback", "") or ""),
        "project_score": float(project_eval.get("project_score", 0.0) or 0.0),
        "project_feedback": str(project_eval.get("project_feedback", "") or ""),
        "overall_summary": str(summary.get("overall_summary", "") or ""),
        "job_key": job_key,
    }

    logger.info(f"Final combined result:\n{json.dumps(result, indent=2)}")
    logger.info("=== Evaluation job completed ===\n")
    return result
