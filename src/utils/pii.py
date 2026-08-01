"""Rule-based PII and safety pattern detection."""

from __future__ import annotations

import re

PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Small, deliberately conservative default list. Real deployments should
# swap this for a proper classifier (e.g. Perspective API, Detoxify).
TOXIC_TERMS = {
    "idiot", "stupid", "moron", "dumbass", "shut up", "hate you",
    "kill yourself", "worthless", "trash", "garbage human",
}


def find_pii(text: str) -> dict[str, list[str]]:
    """Return a mapping of PII category -> list of matched substrings."""
    text = text or ""
    found: dict[str, list[str]] = {}
    for label, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            found[label] = matches
    return found


def toxicity_hits(text: str) -> list[str]:
    lowered = (text or "").lower()
    return [term for term in TOXIC_TERMS if term in lowered]
