# Ask First - Technical Writeup

## 1. Reasoning Approach

The app treats pattern detection as a timeline and evidence problem, not a keyword search problem.

Python owns the deterministic parts:

- Validate that the upload matches the assignment-style JSON shape.
- Ignore hidden/reference/answer keys outside the user conversation data.
- Normalize users and sessions into Pydantic models.
- Sort each user's sessions into a chronological timeline.
- Package all source text, follow-ups, tags, severity, and assistant responses for the model.
- Validate the model output and reject patterns that cite unknown sessions.
- Request a strict JSON schema from the model when supported, then validate again with Pydantic.

The LLM owns the reasoning step:

- Infer symptoms, triggers, lifestyle changes, improvements, worsening, and repeated episodes from the uploaded sessions.
- Connect events only when the dates and ordering support the connection.
- Explain temporal reasoning, including delays, repeated examples, and intervention response.
- Check counter-evidence and possible alternative explanations.
- Produce a visible reasoning trace that shows observation, temporal direction, recurrence or delay, intervention/dose response when present, counter-evidence, and confidence.
- Use cautious non-diagnostic wording.

Confidence is handled in two layers. First, the model explains confidence using evidence and counter-evidence. Second, a generic confidence guard caps scores that are too strong for the number of supporting sessions or the amount of uncertainty.

The reasoning trace is deliberately not private chain-of-thought. It is an audit trail made of source-backed checks the user can inspect. The chat interface does not re-run raw pattern detection. It answers from the stored validated analysis JSON, which keeps follow-up answers grounded in the same evidence trace the user can export.

## 2. Failure Cases And Hallucination Risks

The largest risk is a plausible but unsupported pattern. A model can over-connect events that happen close together, especially when there are only one or two sessions. The app reduces that risk by requiring session IDs, dates, evidence traces, and counter-evidence for every pattern.

The second risk is invented evidence. The parser filters out evidence that references session IDs not present in the upload, and Pydantic rejects malformed output. This does not prove every interpretation is correct, but it prevents fake sessions from entering the final JSON.

The third risk is overconfident medical language. The system prompt forbids diagnosis claims and requires language such as "may be linked to", "consistent with", and "possible contributor". The confidence scorer also lowers confidence when support is thin or clear uncertainty is present.

The fourth risk is missing subtle delayed or multi-factor patterns. The prompt asks the model to consider timing, delays, repeated symptoms, improvements after changes, and alternative causes, but a production system would need stronger verification and domain knowledge.

## 3. Improvements With More Time

I would add a verifier pass that separately checks each proposed pattern against the original sessions and explicitly searches for counter-evidence.

I would add timeline visualizations so users can see the sequence of events before reading the pattern explanation.

I would add chunk-and-merge support for very large uploads: analyze date windows per user, merge candidate patterns, then run a final evidence verification pass.

I would add stricter structured outputs with a full JSON schema at the API boundary, plus automated tests that feed adversarial JSON and verify that no hidden/reference sections are used.

I would also add clinician-reviewed safety copy and escalation language for severe symptoms, while keeping the app framed as pattern reasoning and not medical advice.
