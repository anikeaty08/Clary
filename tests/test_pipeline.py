"""Regression tests for the Ask First reasoning pipeline."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.data_loader import DataLoader
from src.llm_client import LLMClient
from src.pattern_detector import PatternDetector


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        with open(ROOT / "exmaple json .json", "r", encoding="utf-8") as handle:
            self.sample_data = json.load(handle)

        self.structure = DataLoader().detect_and_parse(self.sample_data)
        self.detector = PatternDetector(LLMClient(api_key="test"))

    def test_assignment_dataset_loads(self) -> None:
        self.assertEqual(len(self.structure.users), 3)
        self.assertEqual(
            sum(len(items) for items in self.structure.conversations.values()),
            27,
        )

    def test_hidden_reference_sections_are_ignored(self) -> None:
        data = dict(self.sample_data)
        data["hidden_patterns_reference"] = {
            "patterns": [{"title": "This must never become source evidence"}],
        }

        structure = DataLoader().detect_and_parse(data)

        self.assertEqual(len(structure.users), 3)
        self.assertTrue(any("hidden_patterns_reference" in warning for warning in structure.warnings))

    def test_pattern_parser_requires_real_sessions_and_adds_trace(self) -> None:
        user = self.structure.users[0]
        conversations = self.structure.conversations[user.user_id]
        response = json.dumps(
            {
                "patterns": [
                    {
                        "pattern_id": "P1",
                        "user_id": "WRONG_USER",
                        "title": "Repeated behavior may be linked to a later symptom",
                        "confidence": "very high",
                        "confidence_reason": "One supporting session should be lowered by calibration.",
                        "sessions_involved": ["USR001_S01", "DOES_NOT_EXIST"],
                        "temporal_reasoning": "The source session is evaluated in timeline order.",
                        "reasoning_trace": [
                            {
                                "step": "observation",
                                "detail": "USR001_S01 contains the observation.",
                            }
                        ],
                        "evidence_trace": [
                            {
                                "session_id": "USR001_S01",
                                "date": "2026-01-05",
                                "evidence": "Source evidence from the uploaded session.",
                            },
                            {
                                "session_id": "DOES_NOT_EXIST",
                                "date": "2026-01-06",
                                "evidence": "This evidence must be filtered.",
                            },
                        ],
                        "counter_evidence": [],
                    }
                ]
            }
        )

        patterns = self.detector._parse_patterns_response(response, user, conversations)

        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].user_id, "USR001")
        self.assertEqual(patterns[0].sessions_involved, ["USR001_S01"])
        self.assertEqual(len(patterns[0].evidence_trace), 1)
        self.assertEqual(patterns[0].confidence, "low")
        self.assertTrue(patterns[0].reasoning_trace)


if __name__ == "__main__":
    unittest.main()
