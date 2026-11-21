"""Conservative response drafts; this module never performs account actions."""
from __future__ import annotations

GENERIC = (
    "I can help route this question to the right support team. Please use your bank's "
    "official app or verified website for account-specific help. Do not share passwords, "
    "PINs, one-time passcodes, or full card numbers in chat."
)

def suggest_response(category: str, confidence: float) -> str:
    """Return a safe suggestion based on the predicted Banking77 intent."""
    if confidence < 0.55:
        return GENERIC
    key = str(category).strip().lower()
    if any(term in key for term in ("fraud", "scam", "not_recogn", "not recognised", "not recognized")):
        return ("If you suspect an unauthorized transaction, lock the affected card using the official "
                "bank app if available and contact the bank immediately through a verified channel. "
                "Never share credentials or one-time passcodes.")
    if "cash_withdrawal" in key or "cash withdrawal" in key or "cash machine" in key or "atm" in key:
        return ("For a cash withdrawal issue, check the transaction in your official banking app. "
                "If cash was not dispensed or the amount looks wrong, contact verified bank support "
                "with the date, amount, and ATM location. Never share your PIN or OTP.")
    if "cash" in key or "cashback" in key:
        return ("Please review the cash transaction in your official banking app. If it is missing "
                "or incorrect, contact verified bank support with the date and amount. Never share "
                "your PIN or OTP.")
    if "card" in key:
        return ("Please check your card status and controls in the bank's official app. If the card "
                "is missing, delayed, or unrecognized, contact verified bank support. Never share "
                "your PIN, OTP, password, or full card number.")
    return GENERIC

def summarize_conversation(messages: list[str], max_sentences: int = 3) -> str:
    """Small extractive summary: select distinct informative customer statements."""
    import re
    sentences = [s.strip() for msg in messages for s in re.split(r"(?<=[.!?])\s+", str(msg)) if s.strip()]
    selected: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        normalized = re.sub(r"\W+", " ", sentence.lower()).strip()
        if len(normalized) < 12 or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(sentence)
        if len(selected) >= max(1, max_sentences):
            break
    return " ".join(selected) if selected else "No conversation text was provided."
