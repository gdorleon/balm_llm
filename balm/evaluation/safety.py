import re
#@gg
# a little strickt conservative content filter, can be extended with more patterns
DISALLOWED = [
    r"kill\b",         # match 'kill' as a whole word
    r"\bviolence\b",   # match 'violence' as a whole word
    r"\bslur\b"        # match 'slur' as a whole word
]

def is_disallowed_content(text: str) -> bool:
    # Check if the input contains any disallowed patterns
    t = text.lower()
    for pat in DISALLOWED:
        if re.search(pat, t):
            return True
    return False
