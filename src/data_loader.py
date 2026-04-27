"""JSON loader for the Ask First assignment dataset format."""
from __future__ import annotations

import json
from typing import Any, Optional

from .schemas import DetectedConversation, DetectedStructure, DetectedUser


class DataLoader:
    """Load, validate, and normalize assignment-style health conversation JSON.

    The app intentionally supports the provided dataset shape first:

    {
      "dataset_info": {...},
      "users": [
        {
          "user_id": "...",
          "name": "...",
          "conversations": [
            {
              "session_id": "...",
              "timestamp": "...",
              "user_message": "...",
              "user_followup": "...",
              "clary_response": "...",
              "tags": [...]
            }
          ]
        }
      ]
    }

    Extra top-level reference or answer keys are ignored and are never sent to the
    LLM. This keeps the analysis grounded in conversations only.
    """

    USER_ID_FIELDS = ("user_id", "userId", "id")
    NAME_FIELDS = ("name", "user_name", "userName")
    CONVERSATION_FIELDS = ("conversations", "sessions")
    SESSION_ID_FIELDS = ("session_id", "sessionId", "id")
    TIMESTAMP_FIELDS = ("timestamp", "date", "created_at", "createdAt")
    USER_MESSAGE_FIELDS = ("user_message", "message", "content", "text")
    USER_FOLLOWUP_FIELDS = ("user_followup", "followup", "follow_up", "user_follow_up")
    ASSISTANT_FIELDS = ("clary_response", "assistant_response", "assistant", "response")

    def load_json_file(self, file_path: str) -> dict[str, Any]:
        """Load JSON from a file path."""

        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Uploaded JSON must be an object with a users array.")
        return data

    def load_json_string(self, json_str: str) -> dict[str, Any]:
        """Parse JSON from a string."""

        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Uploaded JSON must be an object with a users array.")
        return data

    def detect_and_parse(self, data: dict[str, Any] | list[Any]) -> DetectedStructure:
        """Validate and parse the supported assignment-style JSON."""

        if isinstance(data, list):
            data = {"users": data}
        if not isinstance(data, dict):
            raise ValueError("Uploaded JSON must be an object with a users array.")

        users_data = data.get("users")
        if not isinstance(users_data, list) or not users_data:
            raise ValueError("Unsupported JSON format. Expected a non-empty top-level 'users' array.")

        warnings: list[str] = []
        ignored_keys = [
            key
            for key in data
            if key.lower().endswith("reference") or "hidden" in key.lower() or "answer" in key.lower()
        ]
        if ignored_keys:
            warnings.append(
                "Ignored top-level reference/answer keys: " + ", ".join(sorted(ignored_keys))
            )

        structure = DetectedStructure(
            structure_type="assignment_users",
            dataset_info=data.get("dataset_info", {}) if isinstance(data.get("dataset_info"), dict) else {},
            warnings=warnings,
        )

        seen_users: set[str] = set()
        for user_index, user_data in enumerate(users_data, start=1):
            if not isinstance(user_data, dict):
                raise ValueError(f"User entry #{user_index} must be a JSON object.")

            user_id = self._extract_field(user_data, self.USER_ID_FIELDS)
            if not user_id:
                raise ValueError(f"User entry #{user_index} is missing 'user_id'.")

            user_id = str(user_id)
            if user_id in seen_users:
                raise ValueError(f"Duplicate user_id found: {user_id}")
            seen_users.add(user_id)

            conversations_data = self._extract_conversations(user_data)
            if not conversations_data:
                raise ValueError(f"User {user_id} must include a non-empty conversations array.")

            profile = {
                key: value
                for key, value in user_data.items()
                if key not in self.CONVERSATION_FIELDS
            }
            structure.users.append(
                DetectedUser(
                    user_id=user_id,
                    user_name=self._string_or_none(self._extract_field(user_data, self.NAME_FIELDS)),
                    profile=profile,
                )
            )
            structure.conversations[user_id] = self._parse_conversation_list(
                user_id=user_id,
                conversations=conversations_data,
            )

        return structure

    def _extract_conversations(self, user_data: dict[str, Any]) -> list[Any]:
        """Return the conversations array from a user object."""

        for field in self.CONVERSATION_FIELDS:
            value = user_data.get(field)
            if isinstance(value, list):
                return value
        return []

    def _parse_conversation_list(
        self,
        user_id: str,
        conversations: list[Any],
    ) -> list[DetectedConversation]:
        """Normalize all conversation objects for one user."""

        parsed: list[DetectedConversation] = []
        seen_sessions: set[str] = set()

        for index, conversation in enumerate(conversations, start=1):
            if not isinstance(conversation, dict):
                raise ValueError(f"Conversation #{index} for {user_id} must be a JSON object.")

            session_id = self._extract_field(conversation, self.SESSION_ID_FIELDS)
            if not session_id:
                raise ValueError(f"Conversation #{index} for {user_id} is missing 'session_id'.")

            session_id = str(session_id)
            if session_id in seen_sessions:
                raise ValueError(f"Duplicate session_id found for {user_id}: {session_id}")
            seen_sessions.add(session_id)

            if not self._extract_field(conversation, self.TIMESTAMP_FIELDS):
                raise ValueError(f"Conversation {session_id} for {user_id} is missing 'timestamp'.")

            parsed.append(self._parse_single_conversation(conversation, session_id))

        return parsed

    def _parse_single_conversation(
        self,
        conversation: dict[str, Any],
        session_id: str,
    ) -> DetectedConversation:
        """Normalize one session while preserving raw data for traceability."""

        questions = conversation.get("clary_questions", [])
        if isinstance(questions, str):
            questions = [questions]
        if not isinstance(questions, list):
            questions = []

        tags = conversation.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, list):
            tags = []

        return DetectedConversation(
            session_id=session_id,
            timestamp=self._string_or_none(self._extract_field(conversation, self.TIMESTAMP_FIELDS)),
            user_message=self._string_or_none(self._extract_field(conversation, self.USER_MESSAGE_FIELDS)),
            user_followup=self._string_or_none(self._extract_field(conversation, self.USER_FOLLOWUP_FIELDS)),
            clary_questions=[str(item) for item in questions if str(item).strip()],
            assistant_response=self._string_or_none(self._extract_field(conversation, self.ASSISTANT_FIELDS)),
            tags=[str(item) for item in tags if str(item).strip()],
            severity=self._string_or_none(conversation.get("severity")),
            raw_data=conversation,
        )

    def _extract_field(self, data: dict[str, Any], field_names: tuple[str, ...]) -> Optional[Any]:
        """Extract the first present field value from a JSON object."""

        for field in field_names:
            if field in data:
                return data[field]
        return None

    def _string_or_none(self, value: Any) -> Optional[str]:
        """Convert a JSON scalar to a stripped string when present."""

        if value is None:
            return None
        text = str(value).strip()
        return text or None
