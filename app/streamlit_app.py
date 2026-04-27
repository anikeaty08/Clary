"""Streamlit application for AskFirst Clary health pattern reasoning."""
from __future__ import annotations

import json
import re
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
    page_title="Clary - Health Pattern Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        /* ========================================
           BASE STYLES - CHATGPT STYLE
           ======================================== */
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f7f8fa;
            --bg-tertiary: #ececef;
            --text-primary: #0a0a0a;
            --text-secondary: #6f6f6f;
            --text-muted: #a0a0a0;
            --border-color: #e5e5e5;
            --accent-green: #10b981;
            --accent-purple: #8b5cf6;
            --accent-blue: #3b82f6;
            --accent-pink: #ec4899;
        }

        .stApp {
            background: var(--bg-primary) !important;
            color: var(--text-primary) !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        /* ========================================
           LAYOUT
           ======================================== */
        .app-wrapper {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        /* ========================================
           TOP NAVBAR
           ======================================== */
        .navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 24px;
            background: var(--bg-primary);
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .nav-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .logo-icon svg {
            width: 20px;
            height: 20px;
            color: white;
        }

        .brand-name {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.3px;
        }

        .nav-right {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .nav-btn {
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
            border: none;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .nav-btn-ghost {
            background: transparent;
            color: var(--text-secondary);
        }

        .nav-btn-ghost:hover {
            background: var(--bg-secondary);
            color: var(--text-primary);
        }

        .nav-btn-primary {
            background: var(--accent-purple);
            color: white;
        }

        .nav-btn-primary:hover {
            background: #7c3aed;
        }

        /* ========================================
           MAIN CONTENT
           ======================================== */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            width: 100%;
            max-width: 900px;
            margin: 0 auto;
            padding: 0 24px;
        }

        /* ========================================
           WELCOME / HERO SECTION
           ======================================== */
        .hero-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 60px 24px;
            min-height: 60vh;
        }

        .hero-title {
            font-size: 34px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 12px;
            letter-spacing: -0.5px;
            text-align: center;
        }

        .hero-subtitle {
            font-size: 16px;
            color: var(--text-secondary);
            margin-bottom: 40px;
            text-align: center;
            max-width: 500px;
            line-height: 1.6;
        }

        /* ========================================
           INPUT BAR - CHATGPT STYLE
           ======================================== */
        .input-bar-container {
            width: 100%;
            max-width: 720px;
            margin-bottom: 32px;
        }

        .input-bar {
            display: flex;
            align-items: center;
            gap: 12px;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 12px 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: all 0.2s;
        }

        .input-bar:focus-within {
            border-color: #c0c0c0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        }

        .input-icon {
            width: 20px;
            height: 20px;
            color: var(--text-muted);
            flex-shrink: 0;
        }

        .input-field {
            flex: 1;
            border: none;
            outline: none;
            font-size: 15px;
            color: var(--text-primary);
            background: transparent;
            resize: none;
            min-height: 24px;
            max-height: 120px;
            line-height: 1.5;
        }

        .input-field::placeholder {
            color: var(--text-muted);
        }

        .upload-btn {
            width: 38px;
            height: 38px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.15s;
            flex-shrink: 0;
            background: transparent;
            border: none;
        }

        .upload-btn:hover {
            background: var(--bg-secondary);
            color: var(--text-secondary);
        }

        .send-btn {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            background: var(--text-primary);
            border: none;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
            flex-shrink: 0;
        }

        .send-btn:hover {
            background: #333;
            transform: scale(1.02);
        }

        .send-btn svg {
            width: 18px;
            height: 18px;
            color: white;
        }

        /* ========================================
           SUGGESTIONS
           ======================================== */
        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            max-width: 600px;
        }

        .suggestion {
            padding: 10px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            font-size: 13px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.15s;
        }

        .suggestion:hover {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }

        /* ========================================
           STEP INDICATOR
           ======================================== */
        .steps-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 24px 0;
            margin-bottom: 16px;
        }

        .step {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            background: var(--bg-secondary);
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
            color: var(--text-muted);
            transition: all 0.3s;
        }

        .step.active {
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            color: white;
        }

        .step.completed {
            background: var(--accent-green);
            color: white;
        }

        .step-number {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: rgba(255,255,255,0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 600;
        }

        .step-connector {
            width: 24px;
            height: 2px;
            background: var(--border-color);
        }

        .step-connector.active {
            background: var(--accent-green);
        }

        /* ========================================
           ANALYSIS SUMMARY
           ======================================== */
        .summary-card {
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
            border: 1px solid #bbf7d0;
            border-radius: 20px;
            padding: 28px;
            margin-bottom: 24px;
            text-align: center;
        }

        .summary-icon {
            width: 48px;
            height: 48px;
            background: var(--accent-green);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 16px;
        }

        .summary-icon svg {
            width: 24px;
            height: 24px;
            color: white;
        }

        .summary-title {
            font-size: 20px;
            font-weight: 700;
            color: #065f46;
            margin-bottom: 20px;
        }

        .summary-stats {
            display: flex;
            justify-content: center;
            gap: 48px;
        }

        .stat-box {
            text-align: center;
        }

        .stat-number {
            font-size: 42px;
            font-weight: 800;
            color: #059669;
            line-height: 1;
        }

        .stat-label {
            font-size: 12px;
            color: #047857;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 6px;
        }

        /* ========================================
           CHAT AREA
           ======================================== */
        .chat-area {
            display: flex;
            flex-direction: column;
            gap: 20px;
            padding: 24px 0;
            padding-bottom: 100px;
        }

        .chat-message {
            display: flex;
            gap: 14px;
            animation: msgIn 0.3s ease;
        }

        @keyframes msgIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .chat-message.user {
            flex-direction: row-reverse;
        }

        .chat-avatar {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 700;
            flex-shrink: 0;
        }

        .chat-avatar.user {
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }

        .chat-avatar.assistant {
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            color: white;
        }

        .chat-bubble {
            padding: 14px 18px;
            border-radius: 16px;
            font-size: 15px;
            line-height: 1.65;
            max-width: 85%;
        }

        .chat-message.user .chat-bubble {
            background: var(--bg-secondary);
            color: var(--text-primary);
            border-bottom-right-radius: 4px;
        }

        .chat-message.assistant .chat-bubble {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            border-bottom-left-radius: 4px;
        }

        .chat-bubble p {
            margin: 0 0 10px;
        }

        .chat-bubble p:last-child {
            margin-bottom: 0;
        }

        .chat-bubble code {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            background: var(--bg-tertiary);
            padding: 2px 6px;
            border-radius: 4px;
        }

        .chat-bubble pre {
            background: #1a1a2e;
            color: #e5e7eb;
            padding: 14px;
            border-radius: 10px;
            overflow-x: auto;
            margin: 10px 0;
            font-size: 13px;
        }

        .chat-bubble pre code {
            background: transparent;
            padding: 0;
            color: inherit;
        }

        /* ========================================
           TYPING INDICATOR
           ======================================== */
        .typing {
            display: flex;
            gap: 4px;
            padding: 6px 0;
        }

        .typing span {
            width: 7px;
            height: 7px;
            background: var(--accent-purple);
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out;
        }

        .typing span:nth-child(1) { animation-delay: -0.32s; }
        .typing span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes typing {
            0%, 80%, 100% { transform: scale(0.7); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }

        /* ========================================
           FLOATING INPUT
           ======================================== */
        .floating-input {
            position: fixed;
            bottom: 28px;
            left: 50%;
            transform: translateX(-50%);
            width: calc(100% - 48px);
            max-width: 800px;
            z-index: 50;
        }

        /* ========================================
           PATTERN CARDS (Clary style)
           ======================================== */
        .patterns-section {
            margin: 32px 0;
        }

        .section-title {
            font-size: 18px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .pattern-card {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 16px;
            border-left: 4px solid var(--accent-green);
            transition: all 0.15s;
        }

        .pattern-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        }

        .pattern-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            margin-bottom: 12px;
            gap: 12px;
        }

        .pattern-title {
            font-size: 16px;
            font-weight: 700;
            color: var(--text-primary);
        }

        .pattern-badges {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        .badge {
            padding: 4px 10px;
            border-radius: 16px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        .badge-high { background: #d1fae5; color: #065f46; }
        .badge-medium { background: #fef3c7; color: #92400e; }
        .badge-low { background: #fee2e2; color: #991b1b; }
        .badge-user { background: var(--bg-tertiary); color: var(--text-secondary); }

        .pattern-description {
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.6;
            margin-bottom: 16px;
        }

        .pattern-metrics {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }

        .metric {
            text-align: center;
            padding: 12px;
            background: var(--bg-secondary);
            border-radius: 10px;
        }

        .metric-label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }

        .metric-value {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-primary);
        }

        .pattern-note {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 12px;
            background: #eff6ff;
            border-radius: 10px;
            font-size: 13px;
            color: #1e40af;
        }

        .pattern-note svg {
            width: 18px;
            height: 18px;
            flex-shrink: 0;
            margin-top: 2px;
        }

        /* ========================================
           LOADING STATE
           ======================================== */
        .loading-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 80px 24px;
            text-align: center;
        }

        .loading-spinner {
            width: 48px;
            height: 48px;
            border: 3px solid var(--border-color);
            border-top-color: var(--accent-purple);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 24px;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .loading-title {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 8px;
        }

        .loading-sub {
            font-size: 14px;
            color: var(--text-muted);
        }

        /* ========================================
           FOOTER
           ======================================== */
        .footer {
            text-align: center;
            padding: 20px;
            font-size: 12px;
            color: var(--text-muted);
        }

        /* ========================================
           RESPONSIVE
           ======================================== */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 26px;
            }

            .summary-stats {
                gap: 24px;
            }

            .stat-number {
                font-size: 32px;
            }

            .pattern-metrics {
                grid-template-columns: repeat(2, 1fr);
            }

            .chat-bubble {
                max-width: 92%;
            }
        }

        /* ========================================
           HIDE STREAMLIT DEFAULTS
           ======================================== */
        #MainMenu { visibility: hidden !important; }
        .stDeployButton { display: none !important; }
        footer { visibility: hidden !important; }
        .stStatusWidget { display: none !important; }

        /* Custom scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    """Initialize session state."""
    defaults = {
        "analysis_result": None,
        "structure": None,
        "chat_messages": [],
        "mode": "welcome",  # welcome, analyzing, results
        "uploaded_data": None,
        "current_step": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_navbar() -> None:
    """Render top navbar."""
    st.markdown(
        """
        <div class="navbar">
            <div class="nav-left">
                <div class="logo-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                    </svg>
                </div>
                <span class="brand-name">Clary</span>
            </div>
            <div class="nav-right">
                <button class="nav-btn nav-btn-ghost" onclick="window.location.reload()">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
                        <path d="M3 3v5h5"/>
                    </svg>
                    New
                </button>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_steps(current: int) -> None:
    """Render step indicator."""
    steps = [
        ("1", "Upload"),
        ("2", "Analyze"),
        ("3", "Results"),
    ]

    st.markdown('<div class="steps-container">', unsafe_allow_html=True)

    for i, (num, label) in enumerate(steps):
        cls = "completed" if current > i else ("active" if current == i else "")
        icon = "✓" if current > i else num

        st.markdown(f'<div class="step {cls}"><span class="step-number">{icon}</span>{label}</div>', unsafe_allow_html=True)

        if i < len(steps) - 1:
            conn_cls = "active" if current > i else ""
            st.markdown(f'<div class="step-connector {conn_cls}"></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_welcome() -> None:
    """Render welcome/hero section."""
    st.markdown(
        """
        <div class="hero-section">
            <h1 class="hero-title">Hey, Ready to analyze?</h1>
            <p class="hero-subtitle">
                Upload your conversation JSON to detect health patterns with AI-powered temporal reasoning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_bar(placeholder: str = "Ask anything or upload a file...", key: str = "main_input") -> str | None:
    """Render the chat-style input bar."""
    return st.chat_input(placeholder, key=key)


def render_upload_modal() -> dict | None:
    """Render upload area."""
    st.markdown(
        """
        <div style="background: var(--bg-secondary); border: 2px dashed var(--border-color); border-radius: 16px; padding: 40px; text-align: center; margin-bottom: 20px;">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#6f6f6f" stroke-width="1.5" style="margin: 0 auto 16px;">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <p style="font-size: 16px; font-weight: 600; color: var(--text-primary); margin: 0 0 8px;">Drop your JSON file here</p>
            <p style="font-size: 14px; color: var(--text-muted); margin: 0;">or click to browse</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("", type=["json"], label_visibility="collapsed", key="file_uploader")

    if uploaded_file:
        try:
            data = json.loads(uploaded_file.getvalue().decode("utf-8"))
            return data
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            return None

    st.markdown(
        """
        <p style="text-align: center; color: var(--text-muted); margin: 20px 0; font-size: 14px;">
            — or paste JSON below —
        </p>
        """,
        unsafe_allow_html=True,
    )

    pasted = st.text_area(
        "",
        height=160,
        label_visibility="collapsed",
        placeholder='{"users": [{"user_id": "USER_001", "conversations": [...]}]}',
        key="pasted_json",
    )

    if pasted.strip():
        try:
            return json.loads(pasted)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            return None

    return None


def render_chat_message(role: str, content: str) -> None:
    """Render a chat message bubble."""
    avatar = "Y" if role == "user" else "C"
    avatar_cls = "user" if role == "user" else "assistant"
    msg_cls = "user" if role == "user" else "assistant"

    # Format content
    content = re.sub(r'```(\w+)?\n(.*?)```', r'<pre><code>\2</code></pre>', content, flags=re.DOTALL)
    content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)
    content = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', content)

    paragraphs = content.split('\n\n')
    formatted = []
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith('<pre>'):
            formatted.append(f'<p>{p}</p>')
        elif p:
            formatted.append(p)
    content = '\n'.join(formatted)

    st.markdown(
        f"""
        <div class="chat-message {msg_cls}">
            <div class="chat-avatar {avatar_cls}">{avatar}</div>
            <div class="chat-bubble">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary(result: AnalysisResult) -> None:
    """Render analysis summary card."""
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M9 12l2 2 4-4"/>
                    <circle cx="12" cy="12" r="10"/>
                </svg>
            </div>
            <h3 class="summary-title">Analysis Complete</h3>
            <div class="summary-stats">
                <div class="stat-box">
                    <div class="stat-number">{result.total_patterns}</div>
                    <div class="stat-label">Patterns Found</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{result.total_users}</div>
                    <div class="stat-label">Users Analyzed</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pattern_card(pattern: HealthPattern) -> None:
    """Render a pattern card in Clary style."""
    conf_cls = "high" if pattern.confidence in ["high", "very-high"] else pattern.confidence

    # Extract cause and effect from description if possible
    cause = "Pattern detected"
    effect = "Health change"
    temporal_gap = "—"
    evidence_count = len(pattern.evidence_trace)

    # Try to parse from temporal reasoning
    reasoning = pattern.temporal_reasoning

    st.markdown(
        f"""
        <div class="pattern-card">
            <div class="pattern-header">
                <h4 class="pattern-title">{pattern.title}</h4>
                <div class="pattern-badges">
                    <span class="badge badge-{conf_cls}">{pattern.confidence.upper()}</span>
                    <span class="badge badge-user">{pattern.user_id}</span>
                </div>
            </div>
            <p class="pattern-description">{reasoning}</p>
            <div class="pattern-metrics">
                <div class="metric">
                    <div class="metric-label">Evidence</div>
                    <div class="metric-value">{evidence_count} sessions</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{pattern.confidence.upper()}</div>
                </div>
            </div>
            <div class="pattern-note">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
                </svg>
                <span>{pattern.confidence_reason}</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("View reasoning & evidence"):
        st.markdown("**Reasoning Trace:**")
        for item in pattern.reasoning_trace:
            st.markdown(f"- `{item.step}`: {item.detail}")

        st.markdown("**Evidence:**")
        for evidence in pattern.evidence_trace:
            st.markdown(f"- `{evidence.session_id}`: {evidence.evidence}")

        if pattern.counter_evidence:
            st.markdown("**Counter-evidence:**")
            for item in pattern.counter_evidence:
                st.markdown(f"- {item}")

    st.markdown("</div>", unsafe_allow_html=True)


def render_patterns(result: AnalysisResult, users: list) -> None:
    """Render all patterns with filter."""
    st.markdown('<div class="patterns-section">', unsafe_allow_html=True)

    st.markdown(
        """
        <h2 class="section-title">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
            </svg>
            Detected Patterns
        </h2>
        """,
        unsafe_allow_html=True,
    )

    filter_options = ["All users"] + [
        f"{u.user_id}" + (f" ({u.user_name})" if u.user_name else "")
        for u in users
    ]
    selected = st.selectbox("Filter by user", filter_options, label_visibility="collapsed", key="user_filter")

    patterns = result.patterns
    if selected != "All users":
        patterns = [p for p in patterns if p.user_id == selected.split(" ")[0]]

    if not patterns:
        st.info("No patterns found for this filter.")
    else:
        for pattern in patterns:
            render_pattern_card(pattern)

    st.markdown("</div>", unsafe_allow_html=True)


def analyze_json(data: dict | list) -> tuple[DetectedStructure, AnalysisResult]:
    """Run pattern detection."""
    loader = DataLoader()
    structure = loader.detect_and_parse(data)

    llm_client = LLMClient()
    if not llm_client.is_configured():
        raise RuntimeError("OpenAI API key not configured.")

    detector = PatternDetector(llm_client)
    all_patterns = []

    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()

    for index, user in enumerate(structure.users, start=1):
        status_text.text(f"Analyzing {user.user_id}...")
        result = detector.analyze_user(user, structure.conversations.get(user.user_id, []))
        all_patterns.extend(result.patterns)
        progress_bar.progress(index / len(structure.users), text=f"Completed {user.user_id}")

    progress_bar.empty()

    analysis_result = AnalysisResult(
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        total_users=len(structure.users),
        total_patterns=len(all_patterns),
        patterns=all_patterns,
    )

    st.session_state.pattern_detector = detector
    return structure, analysis_result


def render_results_view() -> None:
    """Render the results view with chat and patterns."""
    structure = st.session_state.get("structure")
    result = st.session_state.get("analysis_result")
    detector = st.session_state.get("pattern_detector")

    if not all([structure, result]):
        st.session_state.mode = "welcome"
        st.rerun()
        return

    render_steps(2)

    # Summary
    render_summary(result)

    # Chat area
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)

    if not st.session_state.chat_messages:
        render_chat_message(
            "assistant",
            f"Analysis complete! I found **{result.total_patterns} patterns** across **{result.total_users} users**. "
            "Ask me anything about the findings, or scroll down to see the detailed patterns."
        )

    for msg in st.session_state.chat_messages:
        render_chat_message(msg["role"], msg["content"])

    st.markdown('</div>', unsafe_allow_html=True)

    # Floating input
    st.markdown('<div class="floating-input">', unsafe_allow_html=True)
    prompt = st.chat_input("Ask about the patterns...", key="chat_input")
    st.markdown('</div>', unsafe_allow_html=True)

    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                response = detector.stream_chat(prompt, result)
                if isinstance(response, str):
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                else:
                    full = ""
                    for chunk in response:
                        full += chunk
                    st.session_state.chat_messages.append({"role": "assistant", "content": full})
            except Exception as e:
                st.session_state.chat_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

        st.rerun()

    # Patterns section
    st.divider()
    render_patterns(result, structure.users)

    # Footer
    st.markdown(
        """
        <div class="footer">
            Clary can make mistakes. Verify important health information with a professional.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action buttons
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Analysis", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["mode"]:
                    del st.session_state[key]
            st.rerun()
    with col2:
        json_data = result.model_dump_json(indent=2)
        st.download_button(
            "Export JSON",
            json_data,
            f"clary_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )


def main() -> None:
    """Main app."""
    init_state()
    render_navbar()

    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    mode = st.session_state.mode

    if mode == "welcome":
        render_steps(0)
        render_welcome()
        render_upload_modal()

        # Back button if returning
        if st.session_state.get("structure"):
            if st.button("← Continue Previous Analysis", use_container_width=True):
                st.session_state.mode = "results"
                st.rerun()

    elif mode == "analyzing":
        render_steps(1)

        st.markdown(
            """
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <p class="loading-title">Analyzing Patterns...</p>
                <p class="loading-sub">This may take a few moments</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        data = st.session_state.get("uploaded_data")
        if data:
            try:
                structure, result = analyze_json(data)
                st.session_state.structure = structure
                st.session_state.analysis_result = result
                st.session_state.chat_messages = []
                st.session_state.mode = "results"
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
                st.session_state.mode = "welcome"

    elif mode == "results":
        render_results_view()

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
