# Ask First - Clary Health Pattern Reasoning App

Clary is a Streamlit app for the Ask First AI Intern assignment. It reads a JSON dataset of messy health conversations, builds a timeline for each user, detects cross-conversation patterns with temporal reasoning, scores confidence, supports chat over the stored analysis, and exports validated JSON.

The app does **not** hardcode the hidden pattern answer list. Python handles loading, validation, timeline ordering, and output checks. The LLM reasons over the uploaded sessions only.

## What It Shows

- JSON upload, paste, or bundled sample loading
- Dataset validation preview before analysis
- Per-user chronological timeline
- Pattern cards with temporal reasoning
- Visible source-backed reasoning trace
- Evidence trace with session IDs and dates
- Counter-evidence and uncertainty checks
- Confidence scoring
- Chat interface using stored analysis
- Strict JSON export
- Submission-ready export that filters weak one-off candidates while keeping all candidates available

## Tech Stack

| Part | Choice |
| --- | --- |
| UI | Streamlit |
| Backend | Python |
| LLM | OpenAI model from `OPENAI_MODEL` |
| Default model | `gpt-4.1` in `.env.example`; your local `.env` can override it |
| Validation | Pydantic v2 |
| Output | Strict JSON |

## Install

```powershell
cd C:\Users\parth\Desktop\aiask\ask-first
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4.1
```

You can use another OpenAI model if your key supports it. The app reads the model name from `OPENAI_MODEL` and displays it in the UI.

## Run

```powershell
streamlit run app\streamlit_app.py --server.port 8502
```

Open:

[http://localhost:8502](http://localhost:8502)

## Input Format

The app expects assignment-style JSON:

```json
{
  "dataset_info": {
    "version": "1.0"
  },
  "users": [
    {
      "user_id": "USER_001",
      "name": "Example User",
      "conversations": [
        {
          "session_id": "USER_001_S01",
          "timestamp": "2026-01-05T10:00:00",
          "user_message": "User health concern text here.",
          "user_followup": "Follow-up details here.",
          "clary_response": "Assistant response from the original conversation.",
          "severity": "mild",
          "tags": ["optional", "source", "tags"]
        }
      ]
    }
  ]
}
```

Top-level keys that look like hidden references, answer lists, or solution notes are ignored by the loader and are not sent to the model.

## Output Format

The exported JSON follows this shape:

```json
{
  "analysis_timestamp": "2026-04-27T10:00:00+00:00",
  "total_users": 1,
  "total_patterns": 1,
  "patterns": [
    {
      "pattern_id": "P1",
      "user_id": "USER_001",
      "title": "Behavior change may be linked to later symptom pattern",
      "confidence": "medium",
      "confidence_reason": "The pattern is supported by repeated session evidence and plausible timing, but alternatives remain.",
      "sessions_involved": ["USER_001_S01", "USER_001_S03"],
      "temporal_reasoning": "The reported change appears before later symptom reports in the timeline.",
      "reasoning_trace": [
        {
          "step": "observation",
          "detail": "The uploaded sessions show the repeated observation being considered."
        },
        {
          "step": "temporal_direction",
          "detail": "The possible trigger appears before the symptom in the timeline."
        },
        {
          "step": "counter_evidence",
          "detail": "Sessions without the trigger were checked for similar symptoms."
        },
        {
          "step": "confidence",
          "detail": "The confidence score follows from repeated evidence and remaining uncertainty."
        }
      ],
      "evidence_trace": [
        {
          "session_id": "USER_001_S01",
          "date": "2026-01-05",
          "evidence": "Source-backed observation from the uploaded session."
        }
      ],
      "counter_evidence": ["Relevant uncertainty or missing evidence checked."]
    }
  ]
}
```

## Reasoning Pipeline

1. `DataLoader` validates the upload and normalizes users/sessions.
2. `TimelineBuilder` sorts each user history chronologically.
3. `EventExtractor` packages all source text, tags, severity, follow-ups, and Clary responses without keyword answer tables.
4. `PatternDetector` sends one user timeline at a time to the LLM.
5. The system prompt asks for high-value non-overlapping patterns, temporal direction checks, delay/recurrence reasoning, intervention or dose-response evidence, counter-evidence, cautious medical language, and strict JSON.
6. OpenAI structured JSON schema is requested when supported, with JSON mode fallback.
7. Pydantic validates the response and drops evidence that cites unknown sessions.
8. `ConfidenceScorer` calibrates overconfident outputs using generic evidence-count and counter-evidence rules.
9. `pattern_quality.py` separates submission-ready patterns from weaker exploratory candidates for the default view/export.
10. Chat uses only the stored validated analysis JSON, not fresh hardcoded answers.

## Context And Chunking

The app analyzes one user per LLM call. This keeps the context focused, prevents one user's evidence from leaking into another user's result, and makes validation easier. For larger datasets, the next step would be chunking each user by date window, merging candidate patterns, then running a verifier pass.

## Useful UI Notes

- **Patterns tab**: default view shows submission-ready patterns. Switch to all candidates to inspect weak findings.
- **Timeline tab**: shows raw chronological source context per user.
- **Chat tab**: asks questions over stored analysis only.
- **JSON tab**: can export submission-ready patterns or all candidates.

## Tests

```powershell
python -m unittest discover -s tests
python -m compileall app src tests
python -m json.tool "exmaple json .json"
```

## Project Structure

```text
ask-first/
|-- app/
|   `-- streamlit_app.py
|-- src/
|   |-- schemas.py
|   |-- data_loader.py
|   |-- timeline_builder.py
|   |-- event_extractor.py
|   |-- pattern_detector.py
|   |-- confidence_scorer.py
|   |-- pattern_quality.py
|   |-- llm_client.py
|   |-- prompts.py
|   `-- config.py
|-- tests/
|   `-- test_pipeline.py
|-- requirements.txt
|-- README.md
|-- WRITEUP.md
|-- .env.example
`-- .gitignore
```
