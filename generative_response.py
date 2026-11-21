"""Safe response drafting interface.

The brief mentions generative AI. The supplied intent dataset has no approved bank
knowledge base, policy documents, or account context. This module therefore uses
controlled templates instead of inventing answers or calling an external service.
"""
from response_engine import GENERIC, suggest_response

def generate_draft(category: str, confidence: float, urgent: bool = False) -> str:
    """Produce a conservative human-reviewed draft from a predicted intent."""
    if urgent or confidence < 0.55:
        return GENERIC
    return suggest_response(category, confidence)
