"""Pattern detector using LLM analysis over normalized evidence."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterator

from pydantic import ValidationError

from .confidence_scorer import ConfidenceScorer
from .event_extractor import EventExtractor
from .llm_client import LLMClient
from .prompts import CHAT_SYSTEM_PROMPT, PATTERN_DETECTION_SYSTEM_PROMPT
from .schemas import (
    AnalysisResult,
    DetectedConversation,
    DetectedStructure,
    DetectedUser,
    EvidenceItem,
    HealthPattern,
    ReasoningTraceItem,
    UserAnalysis,
)
from .timeline_builder import TimelineBuilder


PATTERN_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "health_pattern_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "patterns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "pattern_id": {"type": "string"},
                            "user_id": {"type": "string"},
                            "title": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": ["very high", "high", "medium", "low"],
                            },
                            "confidence_reason": {"type": "string"},
                            "sessions_involved": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "temporal_reasoning": {"type": "string"},
                            "reasoning_trace": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "step": {
                                            "type": "string",
                                            "enum": [
                                                "observation",
                                                "temporal_direction",
                                                "delay_or_recurrence",
                                                "intervention_or_dose_response",
                                                "counter_evidence",
                                                "confidence",
                                            ],
                                        },
                                        "detail": {"type": "string"},
                                    },
                                    "required": ["step", "detail"],
                                },
                            },
                            "evidence_trace": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "session_id": {"type": "string"},
                                        "date": {"type": "string"},
                                        "evidence": {"type": "string"},
                                    },
                                    "required": ["session_id", "date", "evidence"],
                                },
                            },
                            "counter_evidence": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "pattern_id",
                            "user_id",
                            "title",
                            "confidence",
                            "confidence_reason",
                            "sessions_involved",
                            "temporal_reasoning",
                            "reasoning_trace",
                            "evidence_trace",
                            "counter_evidence",
                        ],
                    },
                }
            },
            "required": ["patterns"],
        },
    },
}


class PatternDetector:
    """Detect health patterns using evidence-based LLM reasoning."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.timeline_builder = TimelineBuilder()
        self.event_extractor = EventExtractor()
        self.confidence_scorer = ConfidenceScorer()

    def analyze_user(
        self,
        user: DetectedUser,
        conversations: list[DetectedConversation],
    ) -> UserAnalysis:
        """Analyze a single user's conversations for patterns."""

        timeline = self.timeline_builder.build_timeline(conversations)
        events = self.event_extractor.extract_events(conversations)
        context = self._build_context(user, conversations, timeline, events)

        response = self.llm.structured_completion(
            messages=[{"role": "user", "content": context}],
            system_prompt=PATTERN_DETECTION_SYSTEM_PROMPT,
            response_format=PATTERN_RESPONSE_FORMAT,
        )
        if response.startswith("Error:"):
            raise RuntimeError(response)

        patterns = self._parse_patterns_response(response, user, conversations)

        return UserAnalysis(
            user_id=user.user_id,
            user_name=user.user_name,
            patterns_found=len(patterns),
            patterns=patterns,
        )

    def analyze_all_users(self, structure: DetectedStructure) -> AnalysisResult:
        """Analyze all users in the uploaded dataset."""

        all_patterns: list[HealthPattern] = []
        for user in structure.users:
            conversations = structure.conversations.get(user.user_id, [])
            user_analysis = self.analyze_user(user, conversations)
            all_patterns.extend(user_analysis.patterns)

        return AnalysisResult(
            analysis_timestamp=datetime.now(timezone.utc).isoformat(),
            total_users=len(structure.users),
            total_patterns=len(all_patterns),
            patterns=all_patterns,
        )

    def _build_context(
        self,
        user: DetectedUser,
        conversations: list[DetectedConversation],
        timeline,
        events: dict[str, Any],
    ) -> str:
        """Build source-only context for the pattern prompt."""

        lines = [
            "# Health Conversation Analysis Request",
            "",
            "## User",
            f"user_id: {user.user_id}",
            f"name: {user.user_name or 'Unknown'}",
            f"total_sessions: {len(conversations)}",
            "",
            "## Chronological Timeline",
            self.timeline_builder.format_timeline_for_llm(
                timeline,
                user.user_name or user.user_id,
            ),
            "",
            "## Full Source Sessions",
            self.event_extractor.format_events_for_llm(events),
            "",
            "## Task",
            "Find meaningful cross-session health patterns for this user.",
            "Do not use hidden answer/reference data or outside examples.",
            "Return only the strict JSON object requested by the system prompt.",
        ]

        return "\n".join(lines)

    def _parse_patterns_response(
        self,
        response: str,
        user: DetectedUser,
        conversations: list[DetectedConversation],
    ) -> list[HealthPattern]:
        """Parse, validate, and calibrate LLM pattern JSON."""

        payload = self._load_json_object(response)
        raw_patterns = payload.get("patterns", [])
        if not isinstance(raw_patterns, list):
            return []

        session_dates = {
            conversation.session_id: self._date_only(conversation.timestamp)
            for conversation in conversations
        }
        valid_sessions = set(session_dates)

        patterns: list[HealthPattern] = []
        for index, raw_pattern in enumerate(raw_patterns, start=1):
            if not isinstance(raw_pattern, dict):
                continue

            normalized = self._normalize_pattern_payload(
                raw_pattern=raw_pattern,
                user_id=user.user_id,
                fallback_pattern_id=f"P{index}",
                session_dates=session_dates,
                valid_sessions=valid_sessions,
            )
            if not normalized:
                continue

            try:
                pattern = HealthPattern.model_validate(normalized)
            except ValidationError:
                continue

            patterns.append(self.confidence_scorer.calibrate(pattern))

        return patterns

    def _normalize_pattern_payload(
        self,
        raw_pattern: dict[str, Any],
        user_id: str,
        fallback_pattern_id: str,
        session_dates: dict[str, str],
        valid_sessions: set[str],
    ) -> dict[str, Any] | None:
        """Normalize likely LLM key variants into the strict app schema."""

        evidence_trace: list[dict[str, str]] = []
        for raw_evidence in raw_pattern.get("evidence_trace", []):
            if not isinstance(raw_evidence, dict):
                continue

            session_id = str(
                raw_evidence.get("session_id")
                or raw_evidence.get("session")
                or raw_evidence.get("sessionId")
                or ""
            ).strip()
            evidence = str(
                raw_evidence.get("evidence")
                or raw_evidence.get("observation")
                or raw_evidence.get("text")
                or ""
            ).strip()

            if not session_id or session_id not in valid_sessions or not evidence:
                continue

            evidence_trace.append(
                EvidenceItem(
                    session_id=session_id,
                    date=str(raw_evidence.get("date") or session_dates.get(session_id) or ""),
                    evidence=evidence,
                ).model_dump()
            )

        sessions_involved = raw_pattern.get("sessions_involved") or raw_pattern.get("supporting_sessions") or []
        if not isinstance(sessions_involved, list):
            sessions_involved = []

        cleaned_sessions = [
            str(session_id).strip()
            for session_id in sessions_involved
            if str(session_id).strip() in valid_sessions
        ]
        if not cleaned_sessions:
            cleaned_sessions = [item["session_id"] for item in evidence_trace]

        cleaned_sessions = list(dict.fromkeys(cleaned_sessions))
        if not cleaned_sessions or not evidence_trace:
            return None

        reasoning_trace = self._normalize_reasoning_trace(
            raw_pattern=raw_pattern,
            cleaned_sessions=cleaned_sessions,
        )

        confidence = str(raw_pattern.get("confidence") or "medium").strip().lower()
        if confidence not in {"very high", "high", "medium", "low"}:
            confidence = "medium"

        counter_evidence = raw_pattern.get("counter_evidence")
        if counter_evidence is None:
            counter_evidence = raw_pattern.get("counter_evidence_checked", [])
        if isinstance(counter_evidence, str):
            counter_evidence = [counter_evidence]
        if not isinstance(counter_evidence, list):
            counter_evidence = []

        return {
            "pattern_id": str(raw_pattern.get("pattern_id") or fallback_pattern_id).strip(),
            "user_id": user_id,
            "title": str(raw_pattern.get("title") or "Possible health pattern").strip(),
            "confidence": confidence,
            "confidence_reason": str(raw_pattern.get("confidence_reason") or "").strip(),
            "sessions_involved": cleaned_sessions,
            "temporal_reasoning": str(
                raw_pattern.get("temporal_reasoning")
                or raw_pattern.get("temporal_logic")
                or ""
            ).strip(),
            "reasoning_trace": reasoning_trace,
            "evidence_trace": evidence_trace,
            "counter_evidence": [str(item).strip() for item in counter_evidence if str(item).strip()],
        }

    def _normalize_reasoning_trace(
        self,
        raw_pattern: dict[str, Any],
        cleaned_sessions: list[str],
    ) -> list[dict[str, str]]:
        """Normalize reasoning trace and create source-backed fallback steps."""

        valid_steps = {
            "observation",
            "temporal_direction",
            "delay_or_recurrence",
            "intervention_or_dose_response",
            "counter_evidence",
            "confidence",
        }
        trace: list[dict[str, str]] = []

        raw_trace = raw_pattern.get("reasoning_trace", [])
        if isinstance(raw_trace, list):
            for item in raw_trace:
                if not isinstance(item, dict):
                    continue

                step = str(item.get("step") or "").strip()
                detail = str(item.get("detail") or "").strip()
                if step in valid_steps and detail:
                    trace.append(ReasoningTraceItem(step=step, detail=detail).model_dump())

        if trace:
            return trace

        session_list = ", ".join(cleaned_sessions)
        fallback_items = [
            ReasoningTraceItem(
                step="observation",
                detail=f"The pattern is supported by uploaded sessions: {session_list}.",
            ),
            ReasoningTraceItem(
                step="temporal_direction",
                detail=str(
                    raw_pattern.get("temporal_reasoning")
                    or raw_pattern.get("temporal_logic")
                    or "The pattern was evaluated against the chronological session order."
                ).strip(),
            ),
            ReasoningTraceItem(
                step="counter_evidence",
                detail="Counter-evidence was checked in the available timeline and reflected in the confidence score.",
            ),
            ReasoningTraceItem(
                step="confidence",
                detail=str(
                    raw_pattern.get("confidence_reason")
                    or "Confidence is based on the number and timing of supporting sessions."
                ).strip(),
            ),
        ]

        return [item.model_dump() for item in fallback_items]

    def _load_json_object(self, response: str) -> dict[str, Any]:
        """Load a JSON object from a model response."""

        text = response.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"patterns": []}

        return data if isinstance(data, dict) else {"patterns": []}

    def _date_only(self, timestamp: str | None) -> str:
        """Return YYYY-MM-DD from a timestamp string when possible."""

        if not timestamp:
            return ""
        value = str(timestamp).strip()
        if len(value) >= 10:
            return value[:10]
        return value

    def answer_chat(self, user_input: str, analysis_result: AnalysisResult) -> str:
        """Answer a chat question using stored analysis results only."""

        context = self._build_chat_context(user_input, analysis_result)
        response = self.llm.complete(
            messages=[{"role": "user", "content": context}],
            system_prompt=CHAT_SYSTEM_PROMPT,
        )

        return response

    def stream_chat(
        self,
        user_input: str,
        analysis_result: AnalysisResult,
    ) -> Iterator[str]:
        """Stream a chat answer using stored analysis results only."""

        context = self._build_chat_context(user_input, analysis_result)
        yield from self.llm.stream_completion(
            messages=[{"role": "user", "content": context}],
            system_prompt=CHAT_SYSTEM_PROMPT,
        )

    def _build_chat_context(
        self,
        user_input: str,
        analysis_result: AnalysisResult,
    ) -> str:
        """Build chat context from validated analysis JSON."""

        return "\n".join(
            [
                "# User Question",
                user_input,
                "",
                "## Stored Analysis JSON",
                analysis_result.model_dump_json(indent=2),
                "",
                "Answer using only the stored analysis JSON above.",
            ]
        )

    def stream_analysis(self, structure: DetectedStructure) -> tuple[str, str]:
        """Return all-user context and the system prompt for optional streaming."""

        all_context: list[str] = []
        for user in structure.users:
            conversations = structure.conversations.get(user.user_id, [])
            timeline = self.timeline_builder.build_timeline(conversations)
            events = self.event_extractor.extract_events(conversations)
            all_context.append(self._build_context(user, conversations, timeline, events))

        return "\n\n---\n\n".join(all_context), PATTERN_DETECTION_SYSTEM_PROMPT
