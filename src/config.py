"""Configuration settings for Ask First"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Pattern detection settings
MAX_PATTERNS_PER_USER = 5
MIN_SESSIONS_FOR_PATTERN = 2

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    "very high": 4,  # 4+ supporting sessions
    "high": 3,       # 3+ supporting sessions
    "medium": 2,     # 2+ supporting sessions
    "low": 1         # 1 session or weak evidence
}