"""Pattern quality helpers for display and export."""
from __future__ import annotations

from .schemas import AnalysisResult, HealthPattern


def confidence_rank(confidence: str) -> int:
    """Return numeric confidence rank for sorting/filtering."""

    return {"very high": 4, "high": 3, "medium": 2, "low": 1}.get(confidence, 0)


def is_submission_ready(pattern: HealthPattern) -> bool:
    """Return whether a pattern should appear in the default submission view.

    This intentionally uses generic evidence quality signals, not hidden pattern
    answers: confidence, number of source sessions, and intervention/reversal
    language in the visible reasoning trace.
    """

    if pattern.confidence == "low":
        return False
    if len(set(pattern.sessions_involved)) >= 2:
        return True

    trace_text = " ".join(item.detail.lower() for item in pattern.reasoning_trace)
    return any(term in trace_text for term in ("improved", "reduced", "resolved", "dose", "rechallenge"))


def filtered_result(result: AnalysisResult, patterns: list[HealthPattern]) -> AnalysisResult:
    """Build a validated result object for filtered export."""

    return AnalysisResult(
        analysis_timestamp=result.analysis_timestamp,
        total_users=result.total_users,
        total_patterns=len(patterns),
        patterns=patterns,
    )
