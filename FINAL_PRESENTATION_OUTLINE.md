# Final presentation outline

Use this as the slide plan for the final project presentation. Fill numerical results from your own run; do not invent performance values.

1. **Title and objective** — AI-Powered Banking Customer Support Intelligence & Response Automation.
2. **Problem statement** — support teams need faster intent routing, consistent drafts, safe escalation, and concise handoffs.
3. **Business use cases** — intent classification, response drafts, confidence routing, conversation summaries, offline analytics.
4. **Dataset and scope** — supplied `train.csv`, `test.csv`, optional `train.xlsx`; `text` input and `category` intent label; show measured rows/classes from `artifacts/data_audit.json`.
5. **Data quality and EDA** — blanks, duplicates, split overlap, class distribution, utterance length, sample review.
6. **System approach** — normalization, word/character TF-IDF, Logistic Regression and calibrated Linear SVM, validation selection, untouched test evaluation.
7. **Results** — validation leaderboard and held-out accuracy, macro/weighted F1, macro precision/recall, top-3 accuracy; include weakest classes and limitations.
8. **Deep learning comparison** — optional BiLSTM result from `python deep_learning_model.py`; disclose if omitted or not runnable.
9. **Confidence and escalation** — high/medium/low thresholds, fraud cue override, human review, no account decisions.
10. **Response and summary demo** — show synthetic query and short conversation; explain drafts are templates and summaries are extractive.
11. **Application and analytics** — Streamlit screenshots and per-intent metrics.
12. **Privacy, risks, and next steps** — no real PII in demos; validate on bank-approved representative data, establish governance, monitor drift, and add approved knowledge grounding before production.

## Evidence to capture

- Data audit JSON and a class-balance chart.
- Validation comparison CSV and final test metric JSON.
- Per-class report rows with the lowest F1 scores.
- Screenshots of high-confidence, ambiguous, and synthetic urgent/fraud cases.
- A sample extractive summary checked against the source conversation.
