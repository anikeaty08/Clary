"""Timeline builder for normalized health conversations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .schemas import DetectedConversation


@dataclass(frozen=True)
class TimelineEvent:
    """A single session-level timeline entry."""

    session_id: str
    date: Optional[datetime]
    event_type: str
    description: str

    @property
    def date_label(self) -> str:
        """Return YYYY-MM-DD when the timestamp can be parsed."""

        return self.date.strftime("%Y-%m-%d") if self.date else "Unknown date"


class TimelineBuilder:
    """Build chronological session timelines without hardcoded health patterns."""

    def build_timeline(self, conversations: list[DetectedConversation]) -> list[TimelineEvent]:
        """Build one normalized event per conversation session."""

        events: list[TimelineEvent] = []
        for conversation in conversations:
            events.append(
                TimelineEvent(
                    session_id=conversation.session_id,
                    date=self._parse_date(conversation.timestamp),
                    event_type="conversation",
                    description=self._summarize_session(conversation),
                )
            )

        events.sort(key=lambda event: event.date if event.date else datetime.min)
        return events

    def _summarize_session(self, conversation: DetectedConversation) -> str:
        """Create a source-backed session summary for the LLM context."""

        parts: list[str] = []
        if conversation.user_message:
            parts.append(f"User message: {conversation.user_message}")
        if conversation.user_followup:
            parts.append(f"User follow-up: {conversation.user_followup}")
        if conversation.tags:
            parts.append(f"Tags: {', '.join(conversation.tags)}")
        if conversation.severity:
            parts.append(f"Severity: {conversation.severity}")

        return " | ".join(parts) if parts else "No conversational text available"

    def _parse_date(self, timestamp: Optional[str]) -> Optional[datetime]:
        """Parse common timestamp strings to datetime."""

        if not timestamp:
            return None

        value = str(timestamp).strip()
        formats = (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
        )

        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def format_timeline_for_llm(
        self,
        events: list[TimelineEvent],
        user_name: str = "User",
    ) -> str:
        """Format the chronological timeline for LLM context."""

        if not events:
            return f"{user_name}: no sessions available."

        lines = [f"# Timeline for {user_name}"]
        for event in events:
            lines.append(f"- {event.date_label} | {event.session_id}: {event.description}")

        return "\n".join(lines)
