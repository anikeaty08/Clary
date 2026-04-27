"""Clary - Health Pattern Detection"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import DataLoader
from src.llm_client import LLMClient
from src.pattern_detector import PatternDetector
from src.schemas import AnalysisResult, DetectedStructure, HealthPattern

st.set_page_config(
    page_title="Clary",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stApp { background: #ffffff; }
        #MainMenu { visibility: hidden; }
        .stDeployButton { display: none; }
        footer { visibility: hidden; }
        [data-testid="stStatusWidget"] { display: none; }
        .main { padding: 0 !important; }
        .element-container { padding: 0.5rem 0; }

        /* Chat bubbles */
        .msg { display: flex; gap: 12px; margin-bottom: 16px; }
        .msg.user { flex-direction: row-reverse; }
        .avatar {
            width: 32px; height: 32px; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-size: 12px; font-weight: 600; flex-shrink: 0;
        }
        .avatar.user { background: #f3f4f6; color: #6b7280; }
        .avatar.assistant { background: #10b981; color: white; }
        .bubble {
            padding: 12px 16px; border-radius: 16px; font-size: 14px;
            line-height: 1.6; max-width: 80%;
        }
        .msg.user .bubble {
            background: #f3f4f6; color: #111;
            border-bottom-right-radius: 4px;
        }
        .msg.assistant .bubble {
            background: #fff; border: 1px solid #e5e7eb;
            color: #111; border-bottom-left-radius: 4px;
        }
        .bubble p { margin: 0 0 8px; }
        .bubble p:last-child { margin: 0; }
        .bubble code {
            font-family: monospace; font-size: 13px;
            background: #f3f4f6; padding: 2px 6px; border-radius: 4px;
        }
        .bubble pre {
            background: #1f2937; color: #f3f4f6;
            padding: 12px; border-radius: 8px; overflow-x: auto;
            margin: 8px 0; font-size: 13px;
        }
        .bubble pre code { background: transparent; padding: 0; }

        /* Summary */
        .summary {
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            border: 1px solid #a7f3d0; border-radius: 12px;
            padding: 20px; margin: 16px 0; text-align: center;
        }
        .summary h3 { color: #065f46; margin: 0 0 12px; font-size: 16px; }
        .stats { display: flex; justify-content: center; gap: 32px; }
        .stat-num { font-size: 32px; font-weight: 700; color: #059669; }
        .stat-label { font-size: 11px; color: #047857; text-transform: uppercase; }

        /* Patterns */
        .pattern {
            background: #fff; border: 1px solid #e5e7eb;
            border-left: 4px solid #10b981; border-radius: 10px;
            padding: 16px; margin: 12px 0;
        }
        .pattern-header { display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px; }
        .pattern-title { font-size: 15px; font-weight: 600; color: #111; margin: 0; }
        .badge {
            padding: 3px 8px; border-radius: 12px;
            font-size: 10px; font-weight: 700; text-transform: uppercase;
        }
        .badge-high { background: #d1fae5; color: #065f46; }
        .badge-medium { background: #fef3c7; color: #92400e; }
        .badge-low { background: #fee2e2; color: #991b1b; }
        .badge-user { background: #f3f4f6; color: #6b7280; }
        .pattern-desc { font-size: 13px; color: #6b7280; margin: 0; line-height: 1.5; }

        /* Loading */
        .loading { text-align: center; padding: 60px 0; }
        .spinner {
            width: 32px; height: 32px; margin: 0 auto 16px;
            border: 3px solid #e5e7eb; border-top-color: #10b981;
            border-radius: 50%; animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading p { color: #6b7280; font-size: 14px; }

        /* Upload */
        .upload-box {
            background: #f9fafb; border: 2px dashed #d1d5db;
            border-radius: 12px; padding: 32px; text-align: center;
        }
        .upload-box:hover { border-color: #10b981; }

        /* Steps */
        .steps {
            display: flex; align-items: center; justify-content: center;
            gap: 8px; padding: 20px 0;
        }
        .step {
            padding: 6px 14px; background: #f3f4f6; border-radius: 16px;
            font-size: 12px; font-weight: 500; color: #9ca3af;
        }
        .step.active { background: #10b981; color: white; }
        .step.done { background: #10b981; color: white; }
        .step-line { width: 20px; height: 2px; background: #e5e7eb; }
        .step-line.done { background: #10b981; }

        /* Chat input */
        .input-wrapper {
            position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
            width: calc(100% - 48px); max-width: 720px; z-index: 100;
        }
        .input-bar {
            display: flex; align-items: center; gap: 8px;
            background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
            padding: 10px 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .input-bar:focus-within {
            border-color: #10b981; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }

        /* Footer */
        .footer { text-align: center; padding: 16px; font-size: 11px; color: #9ca3af; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    defaults = {"mode": "upload", "uploaded_data": None, "analysis_result": None,
                "structure": None, "chat_messages": [], "pattern_detector": None}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def msg(role, content):
    avatar = "Y" if role == "user" else "C"
    cls = "user" if role == "user" else "assistant"
    content = re.sub(r'```\n?(.*?)```', r'<pre><code>\1</code></pre>', content, flags=re.DOTALL)
    content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)
    content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
    paras = [f'<p>{p.strip()}</p>' for p in content.split('\n\n') if p.strip()]
    st.markdown(f'<div class="msg {cls}"><div class="avatar {cls}">{avatar}</div><div class="bubble">{"".join(paras)}</div></div>', unsafe_allow_html=True)


def steps(current):
    html = '<div class="steps">'
    for i, label in enumerate(["Upload", "Analyze", "Results"]):
        cls = "done" if current > i+1 else ("active" if current == i+1 else "")
        icon = "✓" if current > i+1 else str(i+1)
        html += f'<div class="step {cls}">{icon} {label}</div>'
        if i < 2:
            html += f'<div class="step-line {"done" if current > i+1 else ""}"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def summary(result):
    st.markdown(f'''
        <div class="summary">
            <h3>Analysis Complete</h3>
            <div class="stats">
                <div><div class="stat-num">{result.total_patterns}</div><div class="stat-label">Patterns</div></div>
                <div><div class="stat-num">{result.total_users}</div><div class="stat-label">Users</div></div>
            </div>
        </div>
    ''', unsafe_allow_html=True)


def pattern_card(p):
    conf = "high" if p.confidence in ["high","very-high"] else p.confidence
    st.markdown(f'''
        <div class="pattern">
            <div class="pattern-header">
                <h4 class="pattern-title">{p.title}</h4>
                <div>
                    <span class="badge badge-{conf}">{p.confidence}</span>
                    <span class="badge badge-user">{p.user_id}</span>
                </div>
            </div>
            <p class="pattern-desc">{p.temporal_reasoning}</p>
        </div>
    ''', unsafe_allow_html=True)


def analyze(data):
    loader = DataLoader()
    structure = loader.detect_and_parse(data)
    llm = LLMClient()
    if not llm.is_configured():
        raise RuntimeError("OpenAI API key not configured.")
    detector = PatternDetector(llm)
    patterns = []
    prog = st.progress(0, text="Analyzing...")
    for i, user in enumerate(structure.users, 1):
        st.text(f"Analyzing {user.user_id}...")
        res = detector.analyze_user(user, structure.conversations.get(user.user_id, []))
        patterns.extend(res.patterns)
        prog.progress(i / len(structure.users))
    prog.empty()
    result = AnalysisResult(
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        total_users=len(structure.users), total_patterns=len(patterns), patterns=patterns
    )
    st.session_state.pattern_detector = detector
    return structure, result


def main():
    init_state()
    mode = st.session_state.mode

    if mode == "upload":
        steps(1)

        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.caption("Upload JSON")
        uploaded = st.file_uploader("", type=["json"], label_visibility="collapsed")
        if uploaded:
            try:
                st.session_state.uploaded_data = json.loads(uploaded.getvalue())
            except:
                st.error("Invalid JSON")

        st.markdown('<p style="text-align:center;color:#9ca3af;margin:16px 0;">— or paste —</p>', unsafe_allow_html=True)
        pasted = st.text_area("", height=120, label_visibility="collapsed", placeholder='{"users": [...]}')
        if pasted:
            try:
                st.session_state.uploaded_data = json.loads(pasted)
            except:
                st.error("Invalid JSON")

        if st.button("Analyze", type="primary", use_container_width=True):
            if st.session_state.uploaded_data:
                st.session_state.mode = "analyzing"
                st.rerun()
            else:
                st.warning("Upload or paste JSON first")
        st.markdown('</div>', unsafe_allow_html=True)

    elif mode == "analyzing":
        steps(2)
        st.markdown('<div class="loading"><div class="spinner"></div><p>Analyzing patterns...</p></div>', unsafe_allow_html=True)
        try:
            s, r = analyze(st.session_state.uploaded_data)
            st.session_state.structure = s
            st.session_state.analysis_result = r
            st.session_state.mode = "results"
            st.rerun()
        except Exception as e:
            st.error(str(e))
            st.session_state.mode = "upload"

    elif mode == "results":
        s, r = st.session_state.structure, st.session_state.analysis_result
        d = st.session_state.pattern_detector
        steps(3)
        summary(r)

        # Chat
        st.markdown('<div style="padding-bottom:80px;">', unsafe_allow_html=True)
        if not st.session_state.chat_messages:
            msg("assistant", f"Done! Found **{r.total_patterns} patterns** across **{r.total_users} users**. Ask me anything.")
        for m in st.session_state.chat_messages:
            msg(m["role"], m["content"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-wrapper"><div class="input-bar">', unsafe_allow_html=True)
        prompt = st.chat_input("Ask about patterns...")
        st.markdown('</div></div>', unsafe_allow_html=True)

        if prompt:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.spinner(""):
                try:
                    resp = d.stream_chat(prompt, r)
                    full = resp if isinstance(resp, str) else "".join(resp)
                    st.session_state.chat_messages.append({"role": "assistant", "content": full})
                except Exception as e:
                    st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
            st.rerun()

        # Patterns
        st.divider()
        st.caption(f"Patterns ({r.total_patterns})")
        flt = st.selectbox("Filter", ["All users"] + [f"{u.user_id} ({u.user_name})" if u.user_name else u.user_id for u in s.users], label_visibility="collapsed")
        pats = r.patterns
        if flt != "All users":
            pats = [p for p in pats if p.user_id == flt.split(" ")[0]]
        for p in pats:
            pattern_card(p)

        # Actions
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("New", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
        with col2:
            st.download_button("Export", r.model_dump_json(indent=2), f"clary_{datetime.now().strftime('%Y%m%d')}.json", use_container_width=True)

        st.markdown('<div class="footer">Clary can make mistakes.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()