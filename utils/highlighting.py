# ========= Highlight Helper ==========
def highlight_translated_text(text: str, highlights: List[str]) -> str:
    """Insert <highlight> tags around matched highlight sentences in the text."""
    used = set()
    for hl in highlights:
        pattern = re.escape(hl.strip())
        if not pattern or pattern.lower() in used:
            continue
        regex = re.compile(pattern, re.IGNORECASE)
        text, count = regex.subn(r"<highlight>\g<0></highlight>", text, count=1)
        if count > 0:
            used.add(pattern.lower())
    return text

KEY_TERMS = [
    "bribery", "embezzlement", "nepotism", "corruption", "fraud",
    "abuse of power", "favoritism", "money laundering", "kickback", "cronyism"
]

def highlight_keywords(text: str, terms: List[str]) -> str:
    for term in terms:
        pattern = re.compile(rf"(?<!<highlight>)(\b{re.escape(term)}\b)", re.IGNORECASE)
        text = pattern.sub(r"<highlight>\1</highlight>", text)
    return text
