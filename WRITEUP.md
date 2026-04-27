# Ask First - Technical Writeup

## 1. Reasoning Approach

I treated the assignment as a temporal evidence problem, not a keyword matching problem. The app first turns messy conversations into a reliable timeline per user, then asks the LLM to reason over that timeline and return only structured, source-backed JSON.

The deterministic Python layer does the parts that should not be left to the model:

- Validate the uploaded assignment-style JSON.
- Ignore top-level hidden/reference/answer keys so planted answers are never sent to the model.
- Normalize each user and conversation into Pydantic models.
- Preserve source text from user messages, follow-ups, Clary responses, tags, severity, and timestamps.
- Sort sessions chronologically per user.
- Validate model output and reject patterns that cite unknown sessions.
- Calibrate confidence when evidence is too thin for the model's claimed score.

The LLM layer does the reasoning:

- Infer symptoms, triggers, lifestyle changes, improvements, worsening, recurrence, delays, and possible counter-evidence from the uploaded sessions.
- Connect events only when the timeline supports the direction: the possible trigger must happen before the symptom or improvement it explains.
- Look for repeated patterns, delayed patterns, dose/threshold patterns, reversal after intervention, and one root behavior linked to multiple downstream effects.
- Produce a visible reasoning trace for each pattern. This is not private chain-of-thought; it is a concise audit trail with observation, temporal direction, recurrence/delay, intervention/dose-response when present, counter-evidence, and confidence.
- Use cautious health wording such as "may be linked to", "consistent with", and "possible contributor".

The app analyzes one user per LLM call. That is the context management strategy: each call gets a focused user timeline instead of the entire dataset. This reduces cross-user leakage, keeps prompts smaller, and makes it easier to validate session IDs against the correct user.

The final UI has two views: submission-ready patterns and all candidates. Submission-ready is a generic evidence-quality filter, not a hidden answer list. It hides low-confidence one-off candidates by default while keeping them available for transparency.

## 2. Failure Cases And Hallucination Risks

The biggest risk is over-connecting plausible events. A model may see two events near each other and infer a relationship even when the evidence is thin. The app reduces this risk by requiring session IDs, dates, evidence trace, temporal reasoning, and counter-evidence for every pattern.

The second risk is invented or wrong evidence. The parser filters out evidence that references session IDs not present in that user's uploaded data. Pydantic also rejects malformed pattern objects.

The third risk is noisy extra patterns. The assignment has deliberately planted patterns, but a model may also surface broad lifestyle context or one-off low-confidence observations. The app handles this honestly: it keeps all candidates available, but defaults to a submission-ready view that filters weak candidates.

The fourth risk is overconfident health language. The prompt forbids diagnosis claims and the confidence scorer reduces scores when support is thin or uncertainty is explicit. The app should be read as health pattern reasoning, not medical advice.

The fifth risk is missing subtle delayed patterns. Some patterns require domain knowledge about delayed symptoms or multi-step effects. Prompting helps, but a production system would need a separate verifier and a curated clinical knowledge layer.

## 3. What I Would Improve With More Time

I would add a second LLM verifier pass. The first pass would propose candidate patterns; the verifier would search the source timeline for supporting evidence, counter-evidence, and duplicated umbrella patterns before final export.

I would add a timeline visualization showing trigger and symptom events on a date axis. That would make temporal direction easier to inspect than reading text alone.

I would add larger-dataset chunking: analyze each user's timeline in date windows, merge candidate patterns, then run a final per-user verification pass.

I would add clinician-reviewed safety copy for severe symptoms and escalation cases while keeping the system framed as pattern discovery, not diagnosis.

I would add evaluation scripts that compare model outputs against a private rubric without exposing hidden answers in the repository.
