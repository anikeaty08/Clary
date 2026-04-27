"""LangGraph orchestration for the Clary reasoning pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .llm_client import LLMClient
from .pattern_detector import PatternDetector
from .pattern_quality import confidence_rank, is_submission_ready
from .schemas import AnalysisResult, DetectedStructure, HealthPattern
from .timeline_builder import TimelineBuilder


class ReasoningState(TypedDict, total=False):
    """Shared LangGraph state for the analysis run."""

    structure: DetectedStructure
    timeline_summaries: dict[str, str]
    session_counts: dict[str, int]
    patterns: list[HealthPattern]
    detector: PatternDetector
    ready_patterns_count: int
    result: AnalysisResult
    graph_trace: list[str]


@dataclass
class ReasoningGraphRun:
    """Final output from the LangGraph workflow."""

    result: AnalysisResult
    detector: PatternDetector
    graph_trace: list[str]
    ready_patterns_count: int


class ClaryReasoningGraph:
    """Explicit graph for Clary's reasoning workflow."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.timeline_builder = TimelineBuilder()
        self.app = self._compile()

    def run(self, structure: DetectedStructure) -> ReasoningGraphRun:
        """Run the compiled LangGraph app."""

        final_state = self.app.invoke({"structure": structure})
        result = final_state["result"]
        detector = final_state["detector"]
        graph_trace = final_state.get("graph_trace", [])
        ready_patterns_count = final_state.get("ready_patterns_count", 0)

        return ReasoningGraphRun(
            result=result,
            detector=detector,
            graph_trace=graph_trace,
            ready_patterns_count=ready_patterns_count,
        )

    def _compile(self):
        """Compile the graph nodes and edges."""

        graph = StateGraph(ReasoningState)
        graph.add_node("prepare_timelines", self._prepare_timelines)
        graph.add_node("detect_patterns", self._detect_patterns)
        graph.add_node("verify_patterns", self._verify_patterns)
        graph.add_node("score_and_sort", self._score_and_sort)
        graph.add_node("format_output", self._format_output)

        graph.add_edge(START, "prepare_timelines")
        graph.add_edge("prepare_timelines", "detect_patterns")
        graph.add_edge("detect_patterns", "verify_patterns")
        graph.add_edge("verify_patterns", "score_and_sort")
        graph.add_edge("score_and_sort", "format_output")
        graph.add_edge("format_output", END)

        return graph.compile()

    def _prepare_timelines(self, state: ReasoningState) -> dict:
        """Build chronological source timelines per user."""

        structure = state["structure"]
        timeline_summaries: dict[str, str] = {}
        session_counts: dict[str, int] = {}

        for user in structure.users:
            conversations = structure.conversations.get(user.user_id, [])
            timeline = self.timeline_builder.build_timeline(conversations)
            timeline_summaries[user.user_id] = self.timeline_builder.format_timeline_for_llm(
                timeline,
                user.user_name or user.user_id,
            )
            session_counts[user.user_id] = len(conversations)

        total_sessions = sum(session_counts.values())
        return {
            "timeline_summaries": timeline_summaries,
            "session_counts": session_counts,
            "graph_trace": [
                f"prepare_timelines: sorted {total_sessions} sessions across {len(structure.users)} users."
            ],
        }

    def _detect_patterns(self, state: ReasoningState) -> dict:
        """Run LLM pattern detection one user at a time."""

        structure = state["structure"]
        detector = PatternDetector(self.llm_client)
        patterns: list[HealthPattern] = []
        graph_trace = list(state.get("graph_trace", []))

        for user in structure.users:
            conversations = structure.conversations.get(user.user_id, [])
            user_analysis = detector.analyze_user(user, conversations)
            patterns.extend(user_analysis.patterns)
            graph_trace.append(
                f"detect_patterns: {user.user_id} returned {user_analysis.patterns_found} candidate pattern(s)."
            )

        return {
            "detector": detector,
            "patterns": patterns,
            "graph_trace": graph_trace,
        }

    def _verify_patterns(self, state: ReasoningState) -> dict:
        """Drop any pattern that is not grounded in the user's real sessions."""

        structure = state["structure"]
        graph_trace = list(state.get("graph_trace", []))
        valid_sessions = {
            user.user_id: {
                conversation.session_id
                for conversation in structure.conversations.get(user.user_id, [])
            }
            for user in structure.users
        }

        verified: list[HealthPattern] = []
        dropped = 0
        for pattern in state.get("patterns", []):
            user_sessions = valid_sessions.get(pattern.user_id, set())
            cited_sessions = set(pattern.sessions_involved)
            evidence_sessions = {item.session_id for item in pattern.evidence_trace}
            if cited_sessions and evidence_sessions and cited_sessions <= user_sessions and evidence_sessions <= user_sessions:
                verified.append(pattern)
            else:
                dropped += 1

        graph_trace.append(
            f"verify_patterns: kept {len(verified)} pattern(s), dropped {dropped} ungrounded pattern(s)."
        )
        return {
            "patterns": verified,
            "graph_trace": graph_trace,
        }

    def _score_and_sort(self, state: ReasoningState) -> dict:
        """Sort patterns and count submission-ready findings."""

        graph_trace = list(state.get("graph_trace", []))
        patterns = sorted(
            state.get("patterns", []),
            key=lambda pattern: (
                pattern.user_id,
                -confidence_rank(pattern.confidence),
                pattern.pattern_id,
            ),
        )
        ready_count = sum(1 for pattern in patterns if is_submission_ready(pattern))
        graph_trace.append(
            f"score_and_sort: {ready_count} pattern(s) meet the submission-ready evidence filter."
        )
        return {
            "patterns": patterns,
            "ready_patterns_count": ready_count,
            "graph_trace": graph_trace,
        }

    def _format_output(self, state: ReasoningState) -> dict:
        """Create final validated JSON output."""

        structure = state["structure"]
        graph_trace = list(state.get("graph_trace", []))
        patterns = state.get("patterns", [])
        result = AnalysisResult(
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            total_users=len(structure.users),
            total_patterns=len(patterns),
            patterns=patterns,
        )
        graph_trace.append("format_output: generated final Pydantic-validated JSON result.")

        return {
            "result": result,
            "graph_trace": graph_trace,
        }
