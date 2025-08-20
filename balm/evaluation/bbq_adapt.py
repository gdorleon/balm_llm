import re
from typing import Dict, Any

def normalize_freeform(answer: str, cand_a: str, cand_b: str) -> str:
    a = answer.lower()
    unk_phrases = ["cannot tell", "unknown", "not specified", "no information", "no info", "not enough information"]
    for p in unk_phrases:
        if p in a:
            return "Unknown"
    # basic normalization for gender etc.
    def norm(s):
        s = s.lower().strip()
        s = re.sub(r"[^a-z ]+", "", s)
        return s
    a_norm = norm(cand_a)
    b_norm = norm(cand_b)
    if a_norm and a_norm in norm(answer):
        return "A"
    if b_norm and b_norm in norm(answer):
        return "B"
    # fallback unknown
    return "Unknown"
