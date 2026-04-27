"""Pydantic schemas for Ask First"""
from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
from datetime import datetime


class EvidenceItem(BaseModel):
    """Evidence from a single session supporting a pattern"""
    session: str = Field(description="Session ID")
    date: str = Field(description="Date of the session (YYYY-MM-DD)")
    observation: str = Field(description="Key observation from this session")


class HealthPattern(BaseModel):
    """A detected health pattern with evidence"""
    pattern_id: str = Field(description="Unique pattern identifier (e.g., P1, P2)")
    title: str = Field(description="Short title of the pattern")
    user_id: str = Field(description="User ID this pattern belongs to")
    supporting_sessions: list[str] = Field(description="List of session IDs supporting this pattern")
    evidence_trace: list[EvidenceItem] = Field(description="Detailed evidence from each session")
    temporal_logic: str = Field(description="Explanation of timing relationship between trigger and symptom")
    counter_evidence_checked: list[str] = Field(description="List of counter-evidence checks performed")
    causal_mechanism: str = Field(description="Possible mechanism (use 'may suggest', 'consistent with')")
    confidence: Literal["very high", "high", "medium", "low"] = Field(description="Confidence level")
    confidence_reason: str = Field(description="Why this confidence level was assigned")


class UserAnalysis(BaseModel):
    """Analysis results for a single user"""
    user_id: str
    user_name: Optional[str] = None
    patterns_found: int = 0
    patterns: list[HealthPattern] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Final analysis result with all patterns"""
    analysis_timestamp: str = Field(description="ISO timestamp of analysis")
    total_users: int = Field(description="Number of users analyzed")
    total_patterns_found: int = Field(description="Total patterns found across all users")
    results: list[UserAnalysis] = Field(default_factory=list)


class ChatMessage(BaseModel):
    """A chat message in the interface"""
    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str] = None


class DetectedUser(BaseModel):
    """User data extracted from any JSON structure"""
    user_id: str
    user_name: Optional[str] = None
    raw_data: dict = Field(default_factory=dict)


class DetectedConversation(BaseModel):
    """Conversation data extracted from any JSON structure"""
    session_id: str
    timestamp: Optional[str] = None
    user_message: Optional[str] = None
    assistant_response: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    severity: Optional[str] = None
    raw_data: dict = Field(default_factory=dict)


class DetectedStructure(BaseModel):
    """Information about detected JSON structure"""
    users: list[DetectedUser] = Field(default_factory=list)
    conversations: dict[str, list[DetectedConversation]] = Field(default_factory=dict)
    structure_type: str = "unknown"