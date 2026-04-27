"""Generic confidence checks for detected patterns."""
from __future__ import annotations

from .schemas import ConfidenceLevel, HealthPattern


CONFIDENCE_ORDER: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "very high": 4,
}

ORDER_TO_CONFIDENCE: dict[int, ConfidenceLevel] = {
    1: "low",
    2: "medium",
    3: "high",
    4: "very high",
}


class ConfidenceScorer:
    """Apply evidence-count and counter-evidence sanity checks.

    The LLM still explains confidence in context. This guard prevents obviously
    overconfident outputs, for example one evidence item being labeled very high.
    """

    def calibrate(self, pattern: HealthPattern) -> HealthPattern:
        """Return the pattern with confidence capped by generic evidence rules."""

        max_level = self._max_confidence_from_evidence(pattern)
        current_level = CONFIDENCE_ORDER.get(pattern.confidence, 2)
        calibrated_level = min(current_level, max_level)

        if calibrated_level == current_level:
            return pattern

        calibrated_confidence = ORDER_TO_CONFIDENCE[calibrated_level]
        reason = (
            f"{pattern.confidence_reason} Confidence was calibrated from "
            f"{pattern.confidence!r} to {calibrated_confidence!r} because the "
            "available evidence/counter-evidence does not support a stronger score."
        )

        return pattern.model_copy(
            update={
                "confidence": calibrated_confidence,
                "confidence_reason": reason,
            }
        )

    def _max_confidence_from_evidence(self, pattern: HealthPattern) -> int:
        """Compute an upper bound using generic support strength."""

        support_count = len(set(pattern.sessions_involved))
        evidence_count = len(pattern.evidence_trace)
        count = max(support_count, evidence_count)

        if count <= 1:
            max_level = CONFIDENCE_ORDER["low"]
        elif count == 2:
            max_level = CONFIDENCE_ORDER["medium"]
        elif count == 3:
            max_level = CONFIDENCE_ORDER["high"]
        else:
            max_level = CONFIDENCE_ORDER["very high"]

        if pattern.counter_evidence:
            joined = " ".join(pattern.counter_evidence).lower()
            no_counter_signal = any(
                term in joined
                for term in (
                    "no counter",
                    "no clear counter",
                    "no contradiction",
                    "none found",
                )
            )
            uncertainty_signal = any(
                term in joined
                for term in (
                    "contradict",
                    "alternative",
                    "unclear",
                    "missing",
                    "weak",
                    "limited",
                )
            )
            if uncertainty_signal and not no_counter_signal:
                max_level = min(max_level, CONFIDENCE_ORDER["medium"])

        return max_level
