"""Source event packaging for health conversations."""
from __future__ import annotations

from .schemas import DetectedConversation


class EventExtractor:
    """Prepare source evidence for the LLM without hardcoded hidden patterns.

    This class does not decide which words are symptoms or triggers. It packages
    each uploaded session into structured observations so the system prompt can
    ask the LLM to infer symptoms, triggers, lifestyle changes, improvements,
    worsening, and repeated episodes from the data itself.
    """

    def extract_events(self, conversations: list[DetectedConversation]) -> dict:
        """Return source-backed observations grouped by session."""

        sessions: dict[str, dict] = {}
        for conversation in conversations:
            sessions[conversation.session_id] = {
                "date": conversation.timestamp,
                "user_message": conversation.user_message,
                "user_followup": conversation.user_followup,
                "clary_questions": conversation.clary_questions,
                "clary_response": conversation.assistant_response,
                "tags": conversation.tags,
                "severity": conversation.severity,
                "source_text": conversation.source_text(),
            }

        return {
            "sessions": sessions,
            "session_count": len(sessions),
        }

    def format_events_for_llm(self, events: dict) -> str:
        """Format packaged source observations for an LLM prompt."""

        lines = ["# Source Sessions"]
        sessions = events.get("sessions", {})
        for session_id, session in sessions.items():
            lines.append(f"\n## {session_id}")
            if session.get("date"):
                lines.append(f"Date: {session['date']}")
            if session.get("user_message"):
                lines.append(f"User message: {session['user_message']}")
            if session.get("user_followup"):
                lines.append(f"User follow-up: {session['user_followup']}")
            if session.get("clary_questions"):
                lines.append("Clary questions: " + " | ".join(session["clary_questions"]))
            if session.get("clary_response"):
                lines.append(f"Clary response: {session['clary_response']}")
            if session.get("tags"):
                lines.append("Tags: " + ", ".join(session["tags"]))
            if session.get("severity"):
                lines.append(f"Severity: {session['severity']}")

        return "\n".join(lines)
