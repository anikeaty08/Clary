# Ask First - Health Pattern Reasoning App

Ask First is a Streamlit app that reads assignment-format health conversation JSON, builds per-user timelines, asks an OpenAI model to detect evidence-backed patterns across time, and exports validated JSON.

The app does not hardcode hidden pattern answers. Python validates and normalizes the uploaded data; the LLM reasons over the uploaded sessions only.

## Features

- JSON upload or paste for the provided assignment-style format
- Timeline building by user and session date
- Source packaging for symptoms, triggers, lifestyle changes, improvements, worsening, and repeated episodes
- LLM pattern detection with temporal reasoning and evidence traces
- Confidence calibration using evidence count and counter-evidence
- Visible source-backed reasoning trace for every pattern
- Chat with Clary using stored analysis, not hardcoded answers
- Downloadable strict JSON output validated by Pydantic

## Tech Stack

| Component | Technology |
| --- | --- |
| UI | Streamlit |
| Backend | Python |
| LLM | OpenAI `gpt-4.1` by default, configurable to `gpt-4o` |
| Validation | Pydantic v2 |
| Output | Strict JSON |

## Installation

```bash
git clone <repo-url>
cd ask-first
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set:

```bash
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4.1
```

## Run

```bash
streamlit run app/streamlit_app.py
```

## Supported Input Format

The app expects a JSON object with a top-level `users` array. Each user should include `user_id` and `conversations`; each conversation should include `session_id` and `timestamp`.

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

Top-level hidden/reference/answer keys are ignored by the loader and are not sent to the model.

## Reasoning Pipeline

1. `DataLoader` validates the upload and normalizes users and sessions.
2. `TimelineBuilder` sorts each user's sessions chronologically.
3. `EventExtractor` packages source text, tags, severity, and follow-ups without keyword pattern tables.
4. `PatternDetector` sends the timeline and source sessions to the model with a system prompt that requires temporal reasoning, counter-evidence, cautious health wording, visible reasoning trace, and strict JSON.
5. Pydantic validates the model output and rejects malformed patterns or evidence that references unknown sessions.
6. `ConfidenceScorer` caps overconfident scores when the evidence count or counter-evidence does not support them.

The model boundary uses OpenAI structured JSON schema when supported by the selected model, with a JSON-mode fallback for compatibility.

## Output Format

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
      "confidence_reason": "The pattern is supported by repeated session evidence and plausible timing, but alternative explanations remain.",
      "sessions_involved": ["USER_001_S01", "USER_001_S03"],
      "temporal_reasoning": "The reported change appears before the later symptom reports in the timeline.",
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

## Streaming And Chat

Pattern detection updates progress user-by-user in the UI and uses a complete JSON response per user so the app can validate the full object before showing or exporting it. Chat responses stream token-by-token with `LLMClient.stream_completion`, using only the stored validated analysis JSON as context.

## Context And Chunking

The app analyzes one user at a time. This keeps each model call focused on a single timeline, reduces context size, and prevents patterns from one user leaking into another user's analysis. For larger datasets, the next step would be per-user chunking by date range followed by a merge-and-verify pass.

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
|   |-- llm_client.py
|   |-- prompts.py
|   `-- config.py
|-- requirements.txt
|-- README.md
|-- WRITEUP.md
|-- .env.example
`-- .gitignore
```
