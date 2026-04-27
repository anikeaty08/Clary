"""Streamlit application for Ask First health pattern reasoning."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import DataLoader
from src.llm_client import LLMClient
from src.pattern_detector import PatternDetector
from src.schemas import AnalysisResult, DetectedStructure, HealthPattern


st.set_page_config(
    page_title="Ask First - Health Pattern Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #f8fafc;
            color: #111827;
        }
        .app-header {
            padding: 1rem 0 1.25rem;
            border-bottom: 1px solid #d1d5db;
            margin-bottom: 1.25rem;
        }
        .app-header h1 {
            margin: 0;
            font-size: 2.2rem;
            letter-spacing: 0;
        }
        .app-header p {
            color: #4b5563;
            margin: .35rem 0 0;
        }
        .pattern-card {
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 1rem;
            background: #ffffff;
            margin: .75rem 0;
        }
        .pattern-meta {
            color: #4b5563;
            font-size: .9rem;
        }
        .confidence {
            display: inline-block;
            border-radius: 999px;
            padding: .2rem .6rem;
            font-size: .75rem;
            font-weight: 700;
            text-transform: uppercase;
            background: #e5e7eb;
            color: #111827;
        }
        .confidence-high, .confidence-very-high {
            background: #0f766e;
            color: #ecfeff;
        }
        .confidence-medium {
            background: #92400e;
            color: #fef3c7;
        }
        .confidence-low {
            background: #be123c;
            color: #fee2e2;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    """Initialize Streamlit session state."""

    defaults = {
        "analysis_result": None,
        "structure": None,
        "chat_messages": [],
        "selected_user": "All users",
        "last_loaded_json": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    """Render the app header."""

    st.markdown(
        """
        <div class="app-header">
            <h1>Ask First</h1>
            <p>Health pattern reasoning from uploaded conversation JSON.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_json_input() -> dict[str, Any] | list[Any] | None:
    """Read JSON from an uploaded file or pasted text."""

    st.subheader("Upload JSON")
    col_upload, col_paste = st.columns(2)

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload an assignment-format .json file",
            type=["json"],
            help="Expected shape: top-level users array with conversations per user.",
        )
        if uploaded_file is not None:
            try:
                return json.loads(uploaded_file.getvalue().decode("utf-8"))
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON file: {exc}")
                return None

    with col_paste:
        pasted = st.text_area(
            "Or paste JSON",
            height=180,
            placeholder='{"users": [{"user_id": "USER_001", "conversations": [...]}]}',
        )
        if pasted.strip():
            try:
                return json.loads(pasted)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid pasted JSON: {exc}")
                return None

    return None


def analyze_json(data: dict[str, Any] | list[Any]) -> tuple[DetectedStructure, AnalysisResult]:
    """Validate uploaded JSON and run pattern detection."""

    loader = DataLoader()
    structure = loader.detect_and_parse(data)

    llm_client = LLMClient()
    if not llm_client.is_configured():
        raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY in your environment.")

    detector = PatternDetector(llm_client)
    all_patterns = []
    progress = st.progress(0, text="Preparing user timelines...")
    status_text = st.empty()

    for index, user in enumerate(structure.users, start=1):
        status_text.write(f"Analyzing {user.user_id} ({index}/{len(structure.users)})")
        user_result = detector.analyze_user(
            user,
            structure.conversations.get(user.user_id, []),
        )
        all_patterns.extend(user_result.patterns)
        progress.progress(
            index / len(structure.users),
            text=f"Completed {user.user_id}",
        )

    status_text.empty()
    progress.empty()

    result = AnalysisResult(
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        total_users=len(structure.users),
        total_patterns=len(all_patterns),
        patterns=all_patterns,
    )

    st.session_state.pattern_detector = detector
    return structure, result


def render_user_filter(structure: DetectedStructure) -> None:
    """Render a user filter for detected patterns."""

    options = ["All users"]
    options.extend(
        f"{user.user_id}" + (f" ({user.user_name})" if user.user_name else "")
        for user in structure.users
    )
    selected = st.selectbox("Filter by user", options=options)
    st.session_state.selected_user = selected.split(" ")[0] if selected != "All users" else selected


def render_pattern_card(pattern: HealthPattern) -> None:
    """Render one pattern and its evidence."""

    with st.container(border=True):
        st.markdown(f"### {pattern.title}")
        st.markdown(
            f"**Confidence:** `{pattern.confidence}` | "
            f"**User:** `{pattern.user_id}` | "
            f"**Sessions:** {len(pattern.sessions_involved)}"
        )
        st.markdown(f"**Temporal reasoning:** {pattern.temporal_reasoning}")
        st.markdown(f"**Confidence reason:** {pattern.confidence_reason}")

        with st.expander("Reasoning trace and evidence"):
            st.markdown("**Reasoning trace**")
            for item in pattern.reasoning_trace:
                st.markdown(f"- `{item.step}`: {item.detail}")

            st.markdown("**Evidence trace**")
            for evidence in pattern.evidence_trace:
                st.markdown(
                    f"- `{evidence.session_id}` ({evidence.date or 'unknown date'}): {evidence.evidence}"
                )

            if pattern.counter_evidence:
                st.markdown("**Counter-evidence / uncertainty checked**")
                for item in pattern.counter_evidence:
                    st.markdown(f"- {item}")


def render_patterns(result: AnalysisResult) -> None:
    """Render detected patterns."""

    selected_user = st.session_state.get("selected_user", "All users")
    patterns = result.patterns
    if selected_user != "All users":
        patterns = [pattern for pattern in patterns if pattern.user_id == selected_user]

    st.subheader("Detected Patterns")
    st.caption(f"{len(patterns)} shown, {result.total_patterns} total across {result.total_users} users.")

    if not patterns:
        st.info("No supported patterns found for this filter.")
        return

    for pattern in patterns:
        render_pattern_card(pattern)


def render_chat(result: AnalysisResult) -> None:
    """Render chat over stored analysis."""

    st.subheader("Chat with Clary")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about the stored patterns...")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    detector = st.session_state.get("pattern_detector")
    if detector is None:
        response = "Please run analysis before chatting."
    else:
        with st.chat_message("assistant"):
            response = st.write_stream(detector.stream_chat(prompt, result))

    st.session_state.chat_messages.append({"role": "assistant", "content": response})
    st.rerun()


def render_json_export(result: AnalysisResult) -> None:
    """Render validated JSON output and download button."""

    st.subheader("Validated JSON Output")
    output = result.model_dump(mode="json")
    json_text = result.model_dump_json(indent=2)
    st.json(output)

    st.download_button(
        "Download JSON",
        data=json_text,
        file_name=f"health_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )


def main() -> None:
    """Run the Streamlit app."""

    init_state()
    render_header()

    data = read_json_input()
    if data is not None:
        if st.button("Analyze JSON", type="primary"):
            with st.spinner("Analyzing patterns from uploaded sessions..."):
                try:
                    structure, result = analyze_json(data)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.session_state.structure = structure
                    st.session_state.analysis_result = result
                    st.session_state.chat_messages = []
                    st.success(
                        f"Analysis complete: {result.total_patterns} patterns across {result.total_users} users."
                    )
                    for warning in structure.warnings:
                        st.warning(warning)

    structure = st.session_state.get("structure")
    result = st.session_state.get("analysis_result")
    if structure and result:
        st.divider()
        render_user_filter(structure)
        render_patterns(result)
        st.divider()
        render_chat(result)
        st.divider()
        render_json_export(result)

        if st.button("Clear analysis"):
            st.session_state.analysis_result = None
            st.session_state.structure = None
            st.session_state.chat_messages = []
            st.rerun()


if __name__ == "__main__":
    main()
