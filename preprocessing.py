# src/preprocessing.py

import re

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def basic_clean(text: str) -> str:
    text = text.replace("\u00a0", " ")  # no-break space
    text = normalize_whitespace(text)
    return text

def preprocess_text(text: str) -> str:
    """
    Hozircha juda sodda versiya.
    Keyinchalik: transliteratsiya (kiril->lotin), maxsus belgilardan tozalash va h.k.
    """
    return basic_clean(text)
