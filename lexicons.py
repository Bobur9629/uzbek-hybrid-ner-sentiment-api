import csv
from pathlib import Path
from typing import Dict, List, Tuple

from .config import (
    EMOJI_LEXICON_PATH,
    EMOJI_ASCII_PATH,
    NORMALIZATION_PAIRS_PATH,
    PERSON_DICT_PATH,
    LOC_DICT_PATH,
    POSITION_DICT_PATH,
    ORG_DICT_PATH,
    PRODUCT_DICT_PATH,
)


def load_emoji_lexicon(path: Path = EMOJI_LEXICON_PATH) -> Dict[str, float]:
    """
    emoji_lexicon.csv: emoji;weight
    """
    lex = {}
    if not path.exists():
        return lex

    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            emo = row["emoji"]
            try:
                w = float(row["weight"])
            except Exception:
                w = 0.0
            lex[emo] = w
    return lex


def load_emoji_ascii_map(path: Path = EMOJI_ASCII_PATH) -> Dict[str, str]:
    """
    emoji_ascii.csv: ascii;emoji (bizda emoji = base_symbol)
    """
    mapping = {}
    if not path.exists():
        return mapping

    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            ascii_v = row["ascii"]
            emoji_base = row["emoji"]
            mapping[ascii_v] = emoji_base
    return mapping


def load_normalization_pairs(path: Path = NORMALIZATION_PAIRS_PATH) -> List[Tuple[str, str]]:
    """
    uz_normalization_pairs.csv: informal;normalized
    """
    pairs = []
    if not path.exists():
        return pairs

    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            informal = row["informal"].strip()
            normalized = row["normalized"].strip()
            if informal and normalized:
                pairs.append((informal, normalized))
    return pairs


def load_word_normalization_dict() -> Dict[str, str]:
    """
    informal -> normalized dictionary (faqat bir so'zli birliklar uchun)
    """
    pairs = load_normalization_pairs()
    d = {}
    for inf, norm in pairs:
        if " " not in inf and " " not in norm:
            d[inf.lower()] = norm
    return d


def _load_txt_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]


def load_person_list(path: Path = PERSON_DICT_PATH) -> List[str]:
    return _load_txt_list(path)


def load_location_list(path: Path = LOC_DICT_PATH) -> List[str]:
    return _load_txt_list(path)


def load_position_list(path: Path = POSITION_DICT_PATH) -> List[str]:
    return _load_txt_list(path)


def load_org_list(path: Path = ORG_DICT_PATH) -> List[str]:
    return _load_txt_list(path)


def load_product_list(path: Path = PRODUCT_DICT_PATH) -> List[str]:
    return _load_txt_list(path)
