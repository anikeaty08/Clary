"""Configuration settings for Ask First"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# Generic confidence thresholds. These are not health-pattern answers; they are
# evidence-count guards used to prevent overconfident output.
CONFIDENCE_THRESHOLDS = {
    "very high": 4,  # 4+ supporting sessions
    "high": 3,       # 3+ supporting sessions
    "medium": 2,     # 2+ supporting sessions
    "low": 1         # 1 session or weak evidence
}
