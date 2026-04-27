"""Streamlit UI for Clary health pattern reasoning."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE_JSON_PATH = ROOT / "exmaple json .json"

from src.config import OPENAI_MODEL
from src.data_loader import DataLoader
from src.llm_client import LLMClient
from src.pattern_quality import confidence_rank, filtered_result, is_submission_ready
from src.reasoning_graph import ClaryReasoningGraph
from src.schemas import AnalysisResult, DetectedStructure, HealthPattern
from src.timeline_builder import TimelineBuilder


st.set_page_config(
    page_title="Clary | Ask First",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --ink: #172033;
            --muted: #64748b;
            --line: #d7dee8;
            --panel: #ffffff;
            --bg: #f6f8fb;
            --teal: #0f766e;
            --blue: #2563eb;
            --amber: #b45309;
            --rose: #be123c;
        }
        .stApp {
            background: var(--bg);
            color: var(--ink);
        }
        #MainMenu, footer, header, [data-testid="stStatusWidget"], .stDeployButton, [data-testid="stToolbar"] {
            visibility: hidden;
            display: none;
        }
        div[data-testid="stAppViewContainer"] {
            background: var(--bg);
        }
        div[data-testid="stHeader"] {
            display: none;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }
        .topbar {
            border-bottom: 1px solid var(--line);
            padding-bottom: 1rem;
            margin-bottom: 1.25rem;
        }
        .brand {
            font-size: 2rem;
            line-height: 1.1;
            font-weight: 750;
            letter-spacing: 0;
            margin: 0;
        }
        .subtitle {
            margin: .35rem 0 0;
            color: var(--muted);
            font-size: .98rem;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: .5rem;
            margin-top: .9rem;
        }
        .pill {
            border: 1px solid var(--line);
            border-radius: 999px;
            background: var(--panel);
            color: var(--muted);
            padding: .28rem .68rem;
            font-size: .8rem;
            font-weight: 650;
        }
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .8rem;
            margin: .8rem 0 1.2rem;
        }
        .metric-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: .9rem 1rem;
        }
        .metric-label {
            color: var(--muted);
            font-size: .72rem;
            text-transform: uppercase;
            font-weight: 750;
            letter-spacing: .02em;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.65rem;
            font-weight: 760;
            margin-top: .15rem;
        }
        .section-note {
            color: var(--muted);
            font-size: .92rem;
            margin: -.25rem 0 1rem;
        }
        div[data-testid="stExpander"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
        }
        div[data-testid="stWidgetLabel"] p,
        label,
        .stFileUploader label,
        .stTextArea label {
            color: var(--ink) !important;
            font-weight: 700 !important;
        }
        div[data-testid="stFileUploaderDropzone"],
        section[data-testid="stFileUploaderDropzone"],
        textarea,
        input {
            background: #ffffff !important;
            color: var(--ink) !important;
            border-color: var(--line) !important;
        }
        div[data-testid="stFileUploaderDropzone"] *,
        section[data-testid="stFileUploaderDropzone"] *,
        div[data-testid="stBaseButton-secondary"] *,
        button[kind="secondary"] * {
            color: var(--ink) !important;
        }
        div[data-testid="stFileUploaderDropzone"] button,
        section[data-testid="stFileUploaderDropzone"] button,
        button[kind="secondary"] {
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
        }
        div[data-testid="stFileUploaderDropzone"] p,
        div[data-testid="stFileUploaderDropzone"] span,
        div[data-testid="stFileUploaderDropzone"] small {
            color: var(--muted) !important;
        }
        /* Text area placeholder styling */
        textarea::placeholder {
            color: var(--muted) !important;
            opacity: 1 !important;
        }
        textarea {
            color: var(--ink) !important;
            font-size: 14px !important;
        }
        .stTextArea textarea {
            color: var(--ink) !important;
            background: #ffffff !important;
        }
        .stTextArea label {
            color: var(--ink) !important;
            font-weight: 700 !important;
        }
        @media (max-width: 760px) {
            .metric-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    """Initialize app state."""

    defaults = {
        "uploaded_data": None,
        "structure": None,
        "analysis_result": None,
        "pattern_detector": None,
        "graph_trace": [],
        "chat_messages": [],
        "selected_user": "All users",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    """Render the app header."""

    st.markdown(
        f"""
        <div class="topbar">
            <h1 class="brand">Clary</h1>
            <p class="subtitle">Cross-conversation health pattern reasoning for the Ask First assignment.</p>
            <div class="pill-row">
                <span class="pill">Model: {OPENAI_MODEL}</span>
                <span class="pill">LangGraph workflow</span>
                <span class="pill">Strict JSON output</span>
                <span class="pill">Temporal reasoning</span>
                <span class="pill">Evidence trace</span>
                <span class="pill">Confidence scoring</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_input_data() -> dict[str, Any] | list[Any] | None:
    """Read JSON from upload or paste area."""

    sample_col, _ = st.columns([0.34, 0.66])
    with sample_col:
        if SAMPLE_JSON_PATH.exists() and st.button("Load bundled sample JSON", use_container_width=True):
            try:
                with open(SAMPLE_JSON_PATH, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception as exc:
                st.error(f"Could not load sample JSON: {exc}")
                return None

    left, right = st.columns([1, 1], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Upload JSON file",
            type=["json"],
            help="Upload your conversation JSON file to analyze health patterns.",
        )
        if uploaded is not None:
            try:
                return json.loads(uploaded.getvalue().decode("utf-8"))
            except json.JSONDecodeError as exc:
                st.error(f"Invalid uploaded JSON: {exc}")
                return None

    with right:
        pasted = st.text_area(
            "Paste JSON here",
            height=180,
            placeholder='Paste your JSON here...\n\nExample: {"users": [{"user_id": "USER_001", "conversations": [...]}]}',
        )
        if pasted.strip():
            try:
                return json.loads(pasted)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid pasted JSON: {exc}")
                return None

    return None


def validate_structure(data: dict[str, Any] | list[Any]) -> DetectedStructure | None:
    """Validate JSON input and show errors."""

    try:
        return DataLoader().detect_and_parse(data)
    except Exception as exc:
        st.error(str(exc))
        return None


def render_structure_preview(structure: DetectedStructure) -> None:
    """Render upload preview before analysis."""

    total_sessions = sum(len(items) for items in structure.conversations.values())
    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-card"><div class="metric-label">Users</div><div class="metric-value">{len(structure.users)}</div></div>
            <div class="metric-card"><div class="metric-label">Sessions</div><div class="metric-value">{total_sessions}</div></div>
            <div class="metric-card"><div class="metric-label">Format</div><div class="metric-value">OK</div></div>
            <div class="metric-card"><div class="metric-label">Output</div><div class="metric-value">JSON</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for warning in structure.warnings:
        st.warning(warning)

    with st.expander("Detected users and session counts", expanded=False):
        for user in structure.users:
            sessions = structure.conversations.get(user.user_id, [])
            label = f"{user.user_id}"
            if user.user_name:
                label += f" ({user.user_name})"
            st.markdown(f"- `{label}`: {len(sessions)} sessions")


def run_analysis(structure: DetectedStructure) -> AnalysisResult:
    """Analyze all users and return validated result JSON."""

    llm = LLMClient()
    if not llm.is_configured():
        raise RuntimeError("OpenAI API key not configured. Set OPENAI_API_KEY in .env.")

    reasoning_graph = ClaryReasoningGraph(llm)
    progress = st.progress(0, text="Running LangGraph reasoning workflow...")
    status = st.empty()

    status.info("LangGraph nodes: prepare timelines -> detect -> verify -> score -> format")
    progress.progress(0.15, text="Preparing graph state...")
    graph_run = reasoning_graph.run(structure)
    progress.progress(1.0, text="LangGraph workflow complete.")

    status.empty()
    progress.empty()
    st.session_state.pattern_detector = graph_run.detector
    st.session_state.graph_trace = graph_run.graph_trace

    return graph_run.result


def render_result_metrics(result: AnalysisResult, structure: DetectedStructure) -> None:
    """Render top-level result metrics."""

    total_sessions = sum(len(items) for items in structure.conversations.values())
    high_count = sum(1 for pattern in result.patterns if pattern.confidence in {"high", "very high"})
    trace_count = sum(len(pattern.reasoning_trace) for pattern in result.patterns)
    ready_count = sum(1 for pattern in result.patterns if is_submission_ready(pattern))

    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-card"><div class="metric-label">Patterns</div><div class="metric-value">{result.total_patterns}</div></div>
            <div class="metric-card"><div class="metric-label">Submission Ready</div><div class="metric-value">{ready_count}</div></div>
            <div class="metric-card"><div class="metric-label">High Confidence</div><div class="metric-value">{high_count}</div></div>
            <div class="metric-card"><div class="metric-label">Sessions Read</div><div class="metric-value">{total_sessions}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"{trace_count} reasoning trace steps generated across all candidates.")


def confidence_color(confidence: str) -> str:
    """Return a stable color for confidence labels."""

    if confidence in {"high", "very high"}:
        return "green"
    if confidence == "medium":
        return "orange"
    return "red"


def render_pattern(pattern: HealthPattern) -> None:
    """Render one detected pattern with all assignment-required details."""

    with st.container(border=True):
        top_left, top_right = st.columns([0.78, 0.22])
        with top_left:
            st.markdown(f"### {pattern.title}")
            st.caption(f"User `{pattern.user_id}` | Pattern `{pattern.pattern_id}`")
        with top_right:
            st.markdown(
                f":{confidence_color(pattern.confidence)}-badge[{pattern.confidence.upper()}]"
            )

        st.markdown("**Temporal reasoning**")
        st.write(pattern.temporal_reasoning)

        st.markdown("**Confidence reason**")
        st.write(pattern.confidence_reason)

        st.markdown("**Evidence preview**")
        for evidence in pattern.evidence_trace[:2]:
            st.markdown(f"- `{evidence.session_id}` ({evidence.date or 'unknown date'}): {evidence.evidence}")
        if len(pattern.evidence_trace) > 2:
            st.caption(f"{len(pattern.evidence_trace) - 2} more evidence item(s) in the full trace below.")

        with st.expander("Reasoning trace", expanded=True):
            for item in pattern.reasoning_trace:
                st.markdown(f"- **{item.step.replace('_', ' ').title()}**: {item.detail}")

        with st.expander("Evidence trace", expanded=False):
            for evidence in pattern.evidence_trace:
                st.markdown(
                    f"- `{evidence.session_id}` ({evidence.date or 'unknown date'}): {evidence.evidence}"
                )

        with st.expander("Counter-evidence and uncertainty", expanded=False):
            if pattern.counter_evidence:
                for item in pattern.counter_evidence:
                    st.markdown(f"- {item}")
            else:
                st.write("No explicit counter-evidence was returned for this pattern.")


def render_patterns_tab(result: AnalysisResult, structure: DetectedStructure) -> None:
    """Render patterns with filters."""

    view_mode = st.radio(
        "View",
        ["Submission-ready patterns", "All candidates"],
        horizontal=True,
        help="Submission-ready hides low-confidence one-off findings but keeps them available in All candidates.",
    )

    options = ["All users"] + [
        f"{user.user_id}" + (f" ({user.user_name})" if user.user_name else "")
        for user in structure.users
    ]
    selected = st.selectbox("User filter", options=options)
    selected_user = selected.split(" ")[0] if selected != "All users" else selected

    confidence_filter = st.selectbox(
        "Confidence filter",
        options=["all", "very high", "high", "medium", "low"],
        index=0,
    )

    patterns = result.patterns
    if view_mode == "Submission-ready patterns":
        patterns = [pattern for pattern in patterns if is_submission_ready(pattern)]
    if selected_user != "All users":
        patterns = [pattern for pattern in patterns if pattern.user_id == selected_user]
    if confidence_filter != "all":
        patterns = [pattern for pattern in patterns if pattern.confidence == confidence_filter]
    patterns = sorted(
        patterns,
        key=lambda pattern: (
            pattern.user_id,
            -confidence_rank(pattern.confidence),
            pattern.pattern_id,
        ),
    )

    if not patterns:
        st.info("No patterns match the current filters.")
        return

    for pattern in patterns:
        render_pattern(pattern)


def render_timeline_tab(structure: DetectedStructure) -> None:
    """Render chronological source timeline."""

    builder = TimelineBuilder()
    for user in structure.users:
        conversations = structure.conversations.get(user.user_id, [])
        timeline = builder.build_timeline(conversations)
        label = f"{user.user_id}"
        if user.user_name:
            label += f" ({user.user_name})"

        with st.expander(label, expanded=False):
            for event in timeline:
                st.markdown(f"- `{event.date_label}` `{event.session_id}`: {event.description}")


def render_chat_tab(result: AnalysisResult) -> None:
    """Render chat over stored analysis."""

    detector = st.session_state.get("pattern_detector")
    st.caption("Chat answers use only the stored validated analysis JSON.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Type a message to chat about patterns...")
    if not prompt:
        return

    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if detector is None:
        response = "Please run analysis before chatting."
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        with st.chat_message("assistant"):
            response = st.write_stream(detector.stream_chat(prompt, result))

    st.session_state.chat_messages.append({"role": "assistant", "content": response})
    st.rerun()


def render_graph_tab() -> None:
    """Render LangGraph execution trace."""

    st.caption("This shows the explicit LangGraph workflow used for this analysis run.")
    st.code(
        "START -> prepare_timelines -> detect_patterns -> verify_patterns -> "
        "score_and_sort -> format_output -> END",
        language="text",
    )

    trace = st.session_state.get("graph_trace", [])
    if not trace:
        st.info("Run analysis to see the LangGraph execution trace.")
        return

    for index, item in enumerate(trace, start=1):
        st.markdown(f"{index}. {item}")


def render_json_tab(result: AnalysisResult) -> None:
    """Render strict JSON output and download control."""

    ready_patterns = [pattern for pattern in result.patterns if is_submission_ready(pattern)]
    export_mode = st.radio(
        "Export",
        ["Submission-ready patterns", "All candidates"],
        horizontal=True,
    )
    export_result = filtered_result(result, ready_patterns) if export_mode.startswith("Submission") else result
    json_text = export_result.model_dump_json(indent=2)

    col_download, col_count = st.columns([0.7, 0.3])
    with col_download:
        st.download_button(
            "Download validated JSON",
            data=json_text,
            file_name=f"clary_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_count:
        st.metric("Exported patterns", export_result.total_patterns)

    st.json(export_result.model_dump(mode="json"))


def render_results(structure: DetectedStructure, result: AnalysisResult) -> None:
    """Render final analysis workspace."""

    render_result_metrics(result, structure)

    tab_patterns, tab_timeline, tab_graph, tab_chat, tab_json = st.tabs(
        ["Patterns", "Timeline", "LangGraph", "Chat", "JSON"]
    )
    with tab_patterns:
        render_patterns_tab(result, structure)
    with tab_timeline:
        render_timeline_tab(structure)
    with tab_graph:
        render_graph_tab()
    with tab_chat:
        render_chat_tab(result)
    with tab_json:
        render_json_tab(result)

    st.divider()
    if st.button("Start over", use_container_width=True):
        for key in [
            "uploaded_data",
            "structure",
            "analysis_result",
            "pattern_detector",
            "graph_trace",
            "chat_messages",
            "selected_user",
        ]:
            st.session_state.pop(key, None)
        st.rerun()


def main() -> None:
    """Run the app."""

    init_state()
    render_header()

    structure = st.session_state.get("structure")
    result = st.session_state.get("analysis_result")
    if structure and result:
        render_results(structure, result)
        return

    st.subheader("Upload Your Data")
    st.markdown(
        '<p class="section-note">Upload or paste your JSON file to detect health patterns with AI-powered reasoning.</p>',
        unsafe_allow_html=True,
    )

    data = parse_input_data()
    if data is not None:
        st.session_state.uploaded_data = data
        detected_structure = validate_structure(data)
        if detected_structure:
            st.session_state.structure = detected_structure

    structure = st.session_state.get("structure")
    if structure:
        render_structure_preview(structure)

        analyze_col, clear_col = st.columns([0.72, 0.28])
        with analyze_col:
            if st.button("Analyze JSON", type="primary", use_container_width=True):
                try:
                    with st.spinner("Running temporal pattern reasoning..."):
                        st.session_state.analysis_result = run_analysis(structure)
                    st.session_state.chat_messages = []
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        with clear_col:
            if st.button("Clear input", use_container_width=True):
                for key in ["uploaded_data", "structure", "analysis_result", "graph_trace", "chat_messages"]:
                    st.session_state.pop(key, None)
                st.rerun()


if __name__ == "__main__":
    main()
