# training/prepare_ner_silver.py

from pathlib import Path
from typing import List, Dict, Tuple
import json

import pandas as pd

from src.config import SENTIMENT_DATA_PATH, PROC_DIR
from src.preprocessing import preprocess_text
from src.ner_rules import NERRules


def text_to_bio(text: str, entities: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Berilgan text va rules-entity ro'yxatidan tokenlar va BIO taglar yasaydi.

    - text: NERRules ga berilgan matn (odatda preprocess_text dan keyingi)
    - entities: NERRules qaytargan ro'yxat:
        [{"text": ..., "start": ..., "end": ..., "type": ...}, ...]

    Natija:
        tokens: ["2024", "yil", "5", "mart", ...]
        tags:   ["B-Date", "I-Date", "B-Date", "I-Date", ...] yoki "O"
    """
    tokens: List[str] = []
    tags: List[str] = []

    if not text:
        return tokens, tags

    # Entitylarni uzunligiga qarab tartiblaymiz (uzunroq span ustuvor)
    ents_sorted = sorted(
        entities,
        key=lambda e: (e["start"], -(e["end"] - e["start"]))
    )

    offset = 0
    current_ent = None  # hozirgi token qaysi entityga tegishli (agar tegishli bo'lsa)

    words = text.split()
    for w in words:
        raw = w
        start = text.find(raw, offset)
        if start == -1:
            # fallback: topilmasa, shunchaki token sifatida qo'shamiz
            start = offset
        end = start + len(raw)
        offset = end

        # Ushbu token qaysi entitylar bilan kesishishini tekshiramiz
        overlapping: List[Dict] = []
        for e in ents_sorted:
            if not (end <= e["start"] or e["end"] <= start):
                overlapping.append(e)

        if overlapping:
            # Eng uzun (yoki birinchi) entityni olamiz
            # (ents_sorted allaqachon tartiblangan)
            ent = overlapping[0]
            ent_type = ent["type"]

            if current_ent is ent:
                tag = f"I-{ent_type}"
            else:
                tag = f"B-{ent_type}"
                current_ent = ent
        else:
            tag = "O"
            current_ent = None

        tokens.append(raw)
        tags.append(tag)

    return tokens, tags


def save_conll(
    path: Path,
    all_records: List[Dict],
) -> None:
    """
    CoNLL uslubida saqlash:

    # sent_id = 0
    # text = ...
    token1  tag1
    token2  tag2
    ...

    (har bir gap orasida bo'sh satr)
    """
    lines: List[str] = []
    for rec in all_records:
        sid = rec["id"]
        text = rec["text"]
        tokens = rec["tokens"]
        tags = rec["tags"]

        lines.append(f"# sent_id = {sid}")
        lines.append(f"# text = {text}")
        for tok, tag in zip(tokens, tags):
            lines.append(f"{tok}\t{tag}")
        lines.append("")  # bo'sh satr

    path.write_text("\n".join(lines), encoding="utf-8")


def save_jsonl(
    path: Path,
    all_records: List[Dict],
) -> None:
    """
    Har bir qatorda JSON yozilgan .jsonl fayl:

    {"id": 0, "text": "...", "tokens": [...], "tags": [...], "entities": [...]}
    """
    with path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    print("SENTIMENT_DATA_PATH:", SENTIMENT_DATA_PATH)
    df = pd.read_csv(SENTIMENT_DATA_PATH)

    assert "text" in df.columns, "CSV faylda 'text' ustuni bo'lishi shart!"

    ner = NERRules()

    out_dir = PROC_DIR / "ner_silver"
    out_dir.mkdir(parents=True, exist_ok=True)

    conll_path = out_dir / "ner_silver_rules.conll"
    jsonl_path = out_dir / "ner_silver_rules.jsonl"

    all_records: List[Dict] = []

    for idx, row in df.iterrows():
        text_orig = str(row["text"])
        if not text_orig or text_orig.lower() == "nan":
            continue

        # Biz pipeline bilan mos bo'lishi uchun preprocess_text ni qo'llaymiz
        text_prep = preprocess_text(text_orig)

        # Rules-based NER
        ents = ner(text_prep)  # [{"text","start","end","type","source"}, ...]

        tokens, tags = text_to_bio(text_prep, ents)

        if not tokens:
            continue

        rec = {
            "id": int(idx),
            "text": text_prep,
            "tokens": tokens,
            "tags": tags,
            "entities": ents,
        }
        all_records.append(rec)

        if (idx + 1) % 500 == 0:
            print(f"{idx + 1} ta satr qayta ishlanmoqda...")

    print(f"Jami gaplar: {len(all_records)}")

    print("CoNLL faylga saqlayapman:", conll_path)
    save_conll(conll_path, all_records)

    print("JSONL faylga saqlayapman:", jsonl_path)
    save_jsonl(jsonl_path, all_records)

    print("Tayyor. NER ML uchun BIO formatli 'silver' korpus yaratildi.")


if __name__ == "__main__":
    main()
