CV_EVAL_PROMPT = """
Compare the candidate CV content against the provided references (job requirements and CV scoring rubric).
Return strict JSON:
{
  "cv_match_rate": <float 0..1>,
  "cv_feedback": "<2-4 short bullets>"
}
"""

PROJECT_EVAL_PROMPT = """
Evaluate the candidate Project Report against the case study brief and project scoring rubric.
Return strict JSON:
{
  "project_score": <float 1..5>,
  "project_feedback": "<2-4 short bullets>"
}
"""

FINAL_SUMMARY_PROMPT = """
Synthesize the CV evaluation and Project evaluation into a 3-5 sentence overall summary.
Return strict JSON:
{
  "overall_summary": "<text>"
}
"""