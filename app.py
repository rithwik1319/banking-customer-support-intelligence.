"""Banking support intelligence demo with human-review safeguards."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from confidence_escalation import assess_query
from generative_response import generate_draft
from response_engine import summarize_conversation

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
st.set_page_config(page_title="Banking Support Intelligence", page_icon="🏦", layout="wide")

st.title("🏦 Banking Customer Support Intelligence")
st.caption("Intent classification, safe response drafts, escalation, summaries, and support analytics")
if not (ART / "metadata.json").exists() or not (ART / "intent_model.joblib").exists():
    st.warning("Model artifacts are missing. Follow the setup and training steps in README.md, then restart the app.")
    st.stop()
meta = json.loads((ART / "metadata.json").read_text(encoding="utf-8"))
model = joblib.load(ART / "intent_model.joblib")
labels = list(meta["labels"])
tabs = st.tabs(["Assist a query", "Summarize a conversation", "Support analytics", "Project notes"])

with tabs[0]:
    st.subheader("Query triage")
    st.info("Demo only. Never enter account numbers, card numbers, passwords, PINs, or OTPs. No messages are sent and no banking actions are performed.")
    query = st.text_area("Customer message", height=130, placeholder="Type a non-sensitive example support question")
    if st.button("Classify and draft", type="primary", disabled=not query.strip()):
        probs = model.predict_proba([query])[0]
        order = probs.argsort()[::-1]
        category, score = labels[int(order[0])], float(probs[order[0]])
        routing = assess_query(query, category, score)
        band = str(routing["band"]).title()
        escalate = bool(routing["escalate"])
        response = generate_draft(category, score, urgent=(band == "Urgent"))
        left, middle, right = st.columns(3)
        left.metric("Predicted intent", category.replace("_", " ").title())
        middle.metric("Model confidence", f"{score:.1%}")
        right.metric("Confidence band", band)
        st.write("**Routing:**", "Human agent review required" if escalate else "High-confidence draft; agent review required before sending")
        if escalate:
            st.warning(str(routing["reason"]))
        st.markdown("**Suggested response draft**")
        st.write(response)
        st.caption("Draft only. A trained employee must verify it before sending. The model does not access customer accounts or execute actions.")
        top = pd.DataFrame({"Intent": [labels[int(i)].replace("_", " ").title() for i in order[:5]],
                            "Score": [float(probs[i]) for i in order[:5]]})
        st.markdown("**Alternative intents**")
        st.dataframe(top, hide_index=True, use_container_width=True)

with tabs[1]:
    st.subheader("Conversation summary")
    st.caption("Extractive local summary for this session; text is not written to project files.")
    conversation = st.text_area("Paste a non-sensitive conversation, one message per line", height=220,
                                key="conversation", placeholder="Customer: My card has not arrived.\nAgent: When did you order it?")
    if st.button("Summarize", disabled=not conversation.strip()):
        lines = [line.strip() for line in conversation.splitlines() if line.strip()]
        st.markdown("**Short summary draft**")
        st.write(summarize_conversation(lines))
        st.caption("Review for accuracy and remove personal data before sharing the summary.")

with tabs[2]:
    st.subheader("Offline test-set analytics")
    metrics_file, report_file = ART / "test_metrics.json", ART / "classification_report.csv"
    if not metrics_file.exists():
        st.info("Run python train_models.py to generate evaluation outputs.")
    else:
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        cols = st.columns(5)
        for col, key, label in zip(cols, ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "top3_accuracy"),
                                   ("Accuracy", "Macro F1", "Weighted F1", "Macro precision", "Top 3 accuracy")):
            col.metric(label, f"{metrics[key]:.1%}")
        st.caption("Metrics are calculated on the supplied test.csv, held out from model fitting. Scores do not guarantee safe handling of real banking requests.")
        if report_file.exists():
            report = pd.read_csv(report_file, index_col=0)
            st.markdown("**Per-intent performance**")
            class_rows = report.loc[~report.index.isin(["accuracy", "macro avg", "weighted avg"])]
            st.dataframe(class_rows.sort_values("f1-score").head(20).style.format(precision=3), use_container_width=True)
        audit_file = ART / "data_audit.json"
        if audit_file.exists():
            audit = json.loads(audit_file.read_text(encoding="utf-8"))
            a, b, c = st.columns(3)
            a.metric("Training rows", f"{audit['train_rows_after_cleanup']:,}")
            b.metric("Test rows", f"{audit['test_rows_after_cleanup']:,}")
            c.metric("Intents", audit["train_categories"])
            st.caption(f"Exact train/test text overlap: {audit['train_test_exact_text_overlap']}. Duplicate train text rows: {audit['duplicate_train_text_rows']}.")
            counts = pd.Series(audit["train_class_counts"], name="Training examples").sort_values()
            st.bar_chart(counts.tail(20))

with tabs[3]:
    st.subheader("Scope and limits")
    st.markdown("""
- Text intent labels route a message; they do not establish customer identity or authorize account access.
- Confidence thresholds are triage heuristics, not safety guarantees. Low confidence and detected fraud cues require a human.
- Response drafts are conservative templates and must be reviewed before sending.
- The conversation summary is extractive and may omit context; review it against the original.
- This classroom project is not connected to a bank, a live LLM, a payment rail, or a customer record system.
- Evaluate by intent and with human review before any real deployment. Apply the bank's privacy, retention, security, and regulatory controls first.
""")
    st.write("Selected baseline:", meta["model"])
    st.write("Train time (UTC):", meta["trained_at_utc"])
