"""Confidence-based routing rules; thresholds are configurable policy heuristics."""
from __future__ import annotations

HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.55
URGENT_CUES = ("fraud", "scam", "stolen", "unauthori", "not recognised", "not recognized", "not_recogn")

def assess_query(text: str, category: str, confidence: float) -> dict[str, object]:
    """Return a routing recommendation, never an automated account decision."""
    context = f"{text} {category}".lower()
    urgent = any(cue in context for cue in URGENT_CUES)
    if urgent:
        return {"band": "urgent", "escalate": True,
                "reason": "Possible fraud or unauthorized activity: send to trained human support immediately."}
    if confidence >= HIGH_THRESHOLD:
        return {"band": "high", "escalate": False,
                "reason": "High model confidence; agent review remains required before any response is sent."}
    if confidence >= MEDIUM_THRESHOLD:
        return {"band": "medium", "escalate": True,
                "reason": "Agent should verify intent before using a response draft."}
    return {"band": "low", "escalate": True,
            "reason": "Low model confidence; route to a human and do not rely on the suggested intent."}
