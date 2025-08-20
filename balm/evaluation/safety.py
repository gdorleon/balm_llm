import re

# Very conservative filter. Extend as needed.
DISALLOWED = [
    r"kill\b",
    r"\bviolence\b",
    r"\bslur\b"
]

def is_disallowed_content(text: str) -> bool:
    t = text.lower()
    for pat in DISALLOWED:
        if re.search(pat, t):
            return True
    return False
