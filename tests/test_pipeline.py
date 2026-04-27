"""Regression tests for the Ask First reasoning pipeline."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from src.data_loader import DataLoader
from src.llm_client import LLMClient
from src.pattern_detector import PatternDetector
from src.pattern_quality import filtered_result, is_submission_ready
from src.reasoning_graph import ClaryReasoningGraph
from src.schemas import AnalysisResult


ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    """Small fake LLM used to test LangGraph orchestration without API calls."""

    def structured_completion(self, messages, system_prompt=None, response_format=None):
        context = messages[0]["content"]
        user_match = re.search(r"user_id: ([A-Z0-9_]+)", context)
        session_match = re.search(r"## ([A-Z0-9_]+_S\d+)", context)
        user_id = user_match.group(1) if user_match else "USER"
        session_id = session_match.group(1) if session_match else f"{user_id}_S01"
        return json.dumps(
            {
                "patterns": [
                    {
                        "pattern_id": "P1",
                        "user_id": user_id,
                        "title": "Single-session candidate may need more evidence",
                        "confidence": "medium",
                        "confidence_reason": "One source session is enough for a candidate but not strong confidence.",
                        "sessions_involved": [session_id],
                        "temporal_reasoning": "The candidate was evaluated against timeline order.",
                        "reasoning_trace": [
                            {
                                "step": "observation",
                                "detail": f"{session_id} contains the source observation.",
                            },
                            {
                                "step": "confidence",
                                "detail": "Confidence should be calibrated because support is thin.",
                            },
                        ],
                        "evidence_trace": [
                            {
                                "session_id": session_id,
                                "date": "2026-01-01",
                                "evidence": "Source-backed test evidence.",
                            }
                        ],
                        "counter_evidence": [],
                    }
                ]
            }
        )

    def complete(self, messages, system_prompt=None):
        return "fake chat response"

    def stream_completion(self, messages, system_prompt=None):
        yield "fake chat response"


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
        self.assertFalse(is_submission_ready(patterns[0]))

    def test_filtered_result_updates_pattern_count(self) -> None:
        user = self.structure.users[0]
        conversations = self.structure.conversations[user.user_id]
        response = json.dumps(
            {
                "patterns": [
                    {
                        "pattern_id": "P1",
                        "user_id": user.user_id,
                        "title": "Repeated behavior may be linked to a later symptom",
                        "confidence": "high",
                        "confidence_reason": "Three sessions support the pattern.",
                        "sessions_involved": ["USR001_S01", "USR001_S04", "USR001_S07"],
                        "temporal_reasoning": "The behavior appears before the symptom in repeated sessions.",
                        "reasoning_trace": [
                            {
                                "step": "observation",
                                "detail": "USR001_S01, USR001_S04, and USR001_S07 contain repeated evidence.",
                            },
                            {
                                "step": "confidence",
                                "detail": "Three source sessions support high confidence.",
                            },
                        ],
                        "evidence_trace": [
                            {
                                "session_id": "USR001_S01",
                                "date": "2026-01-05",
                                "evidence": "Source evidence from session one.",
                            },
                            {
                                "session_id": "USR001_S04",
                                "date": "2026-01-28",
                                "evidence": "Source evidence from session four.",
                            },
                            {
                                "session_id": "USR001_S07",
                                "date": "2026-02-23",
                                "evidence": "Source evidence from session seven.",
                            },
                        ],
                        "counter_evidence": [],
                    }
                ]
            }
        )

        patterns = self.detector._parse_patterns_response(response, user, conversations)
        ready = [pattern for pattern in patterns if is_submission_ready(pattern)]
        result = filtered_result(
            result=AnalysisResult(
                analysis_timestamp="2026-04-27T10:00:00+00:00",
                total_users=1,
                total_patterns=len(patterns),
                patterns=patterns,
            ),
            patterns=ready,
        )

        self.assertEqual(len(ready), 1)
        self.assertEqual(result.total_patterns, 1)

    def test_langgraph_orchestrates_reasoning_pipeline(self) -> None:
        graph = ClaryReasoningGraph(FakeLLM())

        run = graph.run(self.structure)

        self.assertEqual(run.result.total_users, 3)
        self.assertEqual(run.result.total_patterns, 3)
        self.assertTrue(any("prepare_timelines" in item for item in run.graph_trace))
        self.assertTrue(any("detect_patterns" in item for item in run.graph_trace))
        self.assertTrue(any("verify_patterns" in item for item in run.graph_trace))
        self.assertTrue(any("format_output" in item for item in run.graph_trace))


if __name__ == "__main__":
    unittest.main()
