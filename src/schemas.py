"""Pydantic schemas for Ask First."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


ConfidenceLevel = Literal["very high", "high", "medium", "low"]
TraceStep = Literal[
    "observation",
    "temporal_direction",
    "delay_or_recurrence",
    "intervention_or_dose_response",
    "counter_evidence",
    "confidence",
]


class EvidenceItem(BaseModel):
    """Evidence from a single conversation session."""

    session_id: str = Field(description="Session ID from the uploaded JSON")
    date: str = Field(description="Session date in YYYY-MM-DD format when available")
    evidence: str = Field(description="Short source-backed observation from this session")


class ReasoningTraceItem(BaseModel):
    """A visible, source-backed reasoning step for the user."""

    step: TraceStep = Field(description="Type of reasoning check performed")
    detail: str = Field(description="Concise explanation grounded in uploaded sessions")

    @field_validator("detail")
    @classmethod
    def detail_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("detail cannot be empty")
        return value


class HealthPattern(BaseModel):
    """A detected health pattern with traceable evidence."""

    model_config = ConfigDict(extra="forbid")

    pattern_id: str = Field(description="Unique pattern identifier, for example P1")
    user_id: str = Field(description="User ID this pattern belongs to")
    title: str = Field(description="Short cautious title using non-diagnostic wording")
    confidence: ConfidenceLevel = Field(description="Confidence level")
    confidence_reason: str = Field(description="Why this confidence level was assigned")
    sessions_involved: list[str] = Field(description="Session IDs used as evidence")
    temporal_reasoning: str = Field(description="Explanation of timing across sessions")
    reasoning_trace: list[ReasoningTraceItem] = Field(
        description="Visible reasoning checks used to reach the pattern",
    )
    evidence_trace: list[EvidenceItem] = Field(description="Evidence used for this pattern")
    counter_evidence: list[str] = Field(
        default_factory=list,
        description="Counter-evidence, missing evidence, or alternative explanations checked",
    )

    @field_validator("title", "confidence_reason", "temporal_reasoning")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field cannot be empty")
        return value

    @field_validator("sessions_involved")
    @classmethod
    def sessions_must_not_be_empty(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("sessions_involved must contain at least one session_id")
        return cleaned

    @field_validator("evidence_trace")
    @classmethod
    def evidence_must_not_be_empty(cls, value: list[EvidenceItem]) -> list[EvidenceItem]:
        if not value:
            raise ValueError("evidence_trace must contain at least one evidence item")
        return value

    @field_validator("reasoning_trace")
    @classmethod
    def reasoning_trace_must_not_be_empty(
        cls,
        value: list[ReasoningTraceItem],
    ) -> list[ReasoningTraceItem]:
        if not value:
            raise ValueError("reasoning_trace must contain at least one reasoning step")
        return value


class UserAnalysis(BaseModel):
    """Internal per-user analysis summary."""

    user_id: str
    user_name: Optional[str] = None
    patterns_found: int = 0
    patterns: list[HealthPattern] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Final downloadable analysis result."""

    analysis_timestamp: str = Field(description="ISO timestamp of analysis")
    total_users: int = Field(description="Number of users analyzed")
    total_patterns: int = Field(description="Total patterns found across all users")
    patterns: list[HealthPattern] = Field(default_factory=list)

    @property
    def total_patterns_found(self) -> int:
        """Backward-compatible count used by the Streamlit UI."""

        return self.total_patterns

    @property
    def results(self) -> list[UserAnalysis]:
        """Group flat patterns by user for display without changing export JSON."""

        grouped: dict[str, list[HealthPattern]] = {}
        for pattern in self.patterns:
            grouped.setdefault(pattern.user_id, []).append(pattern)

        return [
            UserAnalysis(
                user_id=user_id,
                patterns_found=len(patterns),
                patterns=patterns,
            )
            for user_id, patterns in grouped.items()
        ]


class ChatMessage(BaseModel):
    """A chat message in the interface."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str] = None


class DetectedUser(BaseModel):
    """User data extracted from the supported assignment-style JSON structure."""

    user_id: str
    user_name: Optional[str] = None
    profile: dict[str, Any] = Field(default_factory=dict)


class DetectedConversation(BaseModel):
    """Conversation data normalized from a user session."""

    session_id: str
    timestamp: Optional[str] = None
    user_message: Optional[str] = None
    user_followup: Optional[str] = None
    clary_questions: list[str] = Field(default_factory=list)
    assistant_response: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    severity: Optional[str] = None
    raw_data: dict[str, Any] = Field(default_factory=dict)

    def source_text(self) -> str:
        """Return all conversational text that can be used as source evidence."""

        parts: list[str] = []
        if self.user_message:
            parts.append(f"User message: {self.user_message}")
        if self.user_followup:
            parts.append(f"User follow-up: {self.user_followup}")
        if self.clary_questions:
            parts.append(f"Clary questions: {' | '.join(self.clary_questions)}")
        if self.assistant_response:
            parts.append(f"Clary response: {self.assistant_response}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        if self.severity:
            parts.append(f"Severity: {self.severity}")
        return "\n".join(parts)


class DetectedStructure(BaseModel):
    """Normalized uploaded dataset."""

    users: list[DetectedUser] = Field(default_factory=list)
    conversations: dict[str, list[DetectedConversation]] = Field(default_factory=dict)
    structure_type: str = "assignment_users"
    dataset_info: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
