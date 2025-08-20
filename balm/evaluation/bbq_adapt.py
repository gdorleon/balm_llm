import re
from typing import Dict, Any

def normalize_freeform(answer: str, cand_a: str, cand_b: str) -> str:
    a = answer.lower()

    # Handle generic "unknown" responses early
    unk_phrases = [
        "cannot tell", "unknown", "not specified",
        "no information", "no info", "not enough information"
    ]
    for p in unk_phrases:
        if p in a:
            return "Unknown"

    # Normalize strings to simplify matching (lowercase, remove punctuation)
    def norm(s):
        s = s.lower().strip()
        s = re.sub(r"[^a-z ]+", "", s)
        return s

    a_norm = norm(cand_a)
    b_norm = norm(cand_b)

    # Try to match normalized answer to candidate A or B
    if a_norm and a_norm in norm(answer):
        return "A"
    if b_norm and b_norm in norm(answer):
        return "B"

    # If no clear match, treat as unknown
    return "Unknown"
