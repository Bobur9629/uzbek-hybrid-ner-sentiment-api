import re
from typing import List, Dict, Any

from .lexicons import (
    load_person_list,
    load_location_list,
    load_position_list,
    load_org_list,
    load_product_list,
)

# Leksikonlarni yuklab olamiz...
PERSON_LIST = load_person_list()
LOC_LIST = load_location_list()
POSITION_LIST = load_position_list()
ORG_LIST = load_org_list()
PRODUCT_LIST = load_product_list()

# Tezroq qidirish uchun set
PERSON_SET = set([x.lower() for x in PERSON_LIST])
LOC_SET = set([x.lower() for x in LOC_LIST])
POSITION_SET = set([x.lower() for x in POSITION_LIST])
ORG_SET = set([x.lower() for x in ORG_LIST])
PRODUCT_SET = set([x.lower() for x in PRODUCT_LIST])

# Frazalar (ko'p so'zli nomlar) uchun alohida ro'yxat
ORG_PHRASES = [x for x in ORG_LIST if " " in x.strip()]
PRODUCT_PHRASES = [x for x in PRODUCT_LIST if " " in x.strip()]


def _strip_punct(token: str) -> str:
    return token.strip(",.!?\"'():;«»“”")


def _strip_uz_suffixes(token: str) -> str:
    """
    Juda soddalashtirilgan qo'shimcha olib tashlash:
    Toshkentga -> Toshkent
    Toshkentdan -> Toshkent
    rektori -> rektor
    """
    t = token.lower()

    # Egalik + kelishik qo'shimchalarini oxiridan tekshiramiz
    for suf in ["lari", "ning", "ni", "ga", "da", "dan", "si"]:
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
            break

    return t


class NERRules:
    """
    Rule-based NER (kuchaytirilgan versiya):

    - Person: person_dict.txt
    - Location: loc_dict.txt (+ soddalashtirilgan qo'shimcha kesish)
    - Position: position_dict.txt
    - Organization: org_dict.txt (frazali matching)
    - Product: product_dict.txt (frazali matching)
    - Date: regexlar
    """

    def __init__(self):
        # Date patternlar
        self.date_patterns = [
            re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
            re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b"),
            re.compile(
                r"\b\d{1,2}\s*(yanvar|fevral|mart|aprel|may|iyun|iyul|avgust|sentabr|oktabr|noyabr|dekabr)\b",
                re.IGNORECASE,
            ),
            re.compile(r"\b\d{4}\s*yil\b", re.IGNORECASE),
        ]

    def _match_dates(self, text: str) -> List[Dict[str, Any]]:
        ents = []
        for pat in self.date_patterns:
            for m in pat.finditer(text):
                ents.append({
                    "text": m.group(0),
                    "start": m.start(),
                    "end": m.end(),
                    "type": "Date",
                    "source": "rules",
                })
        return ents

    def _match_single_token_list(
        self,
        text: str,
        vocab_set: set,
        ent_type: str,
        use_suffix_stripping: bool = False,
        require_capital: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Bir so'zli entitylar uchun (Person, Position, Location va hok.)
        require_capital=True bo'lsa, faqat bosh harfi katta bo'lgan so'zlar olinadi
        (masalan, Ali, Safar, Davlat).
        """
        ents = []
        if not vocab_set:
            return ents

        words = text.split()
        offset = 0
        for w in words:
            raw = w
            clean = _strip_punct(raw)

            if not clean:
                offset += len(raw)
                continue

            # Person uchun: faqat bosh harfi katta bo'lsin
            if require_capital and not clean[0].isupper():
                offset += len(raw)
                continue

            norm = clean.lower()
            if use_suffix_stripping:
                norm = _strip_uz_suffixes(norm)

            start = text.find(raw, offset)
            end = start + len(raw)
            offset = end

            if norm in vocab_set:
                ents.append({
                    "text": raw,
                    "start": start,
                    "end": end,
                    "type": ent_type,
                    "source": "rules",
                })
        return ents

    def _match_phrases(self, text: str,
                       phrases: List[str],
                       ent_type: str) -> List[Dict[str, Any]]:
        """
        ORG va PRODUCT kabi ko'p so'zli nomlar uchun substring qidirish.
        """
        ents = []
        if not phrases:
            return ents

        low_text = text.lower()
        for phrase in phrases:
            ph = phrase.strip()
            if not ph:
                continue
            ph_low = ph.lower()

            # Oddiy substring qidirish (bir nechta bo'lsa, hammasini topamiz)
            start = 0
            while True:
                idx = low_text.find(ph_low, start)
                if idx == -1:
                    break

                end_idx = idx + len(ph_low)

                # Haqiqiy substringni original text'dan olamiz
                span_text = text[idx:end_idx]

                ents.append({
                    "text": span_text,
                    "start": idx,
                    "end": end_idx,
                    "type": ent_type,
                    "source": "rules",
                })

                start = end_idx
        return ents

    def __call__(self, text: str) -> List[Dict[str, Any]]:
        ents: List[Dict[str, Any]] = []

        # Person: faqat bosh harfi katta bo'lgan so'zlar (Ali, Safar, Davlat ...)
        ents.extend(self._match_single_token_list(
            text,
            PERSON_SET,
            "Person",
            use_suffix_stripping=False,
            require_capital=True,
        ))

        # Position (masalan, rektor / professor / direktor ...), suffixlarni kesamiz
        ents.extend(self._match_single_token_list(
            text,
            POSITION_SET,
            "Position",
            use_suffix_stripping=True,
            require_capital=False,
        ))

        # Location (Toshkent, Urganch, Samarqand ...) + qo'shimchalar
        ents.extend(self._match_single_token_list(
            text,
            LOC_SET,
            "Location",
            use_suffix_stripping=True,
            require_capital=False,
        ))

        # Organization va Product uchun frazali matching
        ents.extend(self._match_phrases(text, ORG_PHRASES, "Organization"))
        ents.extend(self._match_phrases(text, PRODUCT_PHRASES, "Product"))

        # Date
        ents.extend(self._match_dates(text))

        # start bo'yicha sort qilamiz
        ents = sorted(ents, key=lambda x: x["start"])
        return ents

