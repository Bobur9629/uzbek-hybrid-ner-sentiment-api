# training/generate_synthetic_ner_data.py

from pathlib import Path
from typing import List, Dict, Tuple
import json
import random
from datetime import date, timedelta

from src.config import PROC_DIR

# ==== Sozlamalar ====

NER_DIR = PROC_DIR / "ner_silver"
NER_DIR.mkdir(parents=True, exist_ok=True)

ORIG_JSONL = NER_DIR / "ner_silver_rules.jsonl"

SYN_JSONL = NER_DIR / "ner_silver_synthetic.jsonl"
SYN_CONLL = NER_DIR / "ner_silver_synthetic.conll"

COMB_JSONL = NER_DIR / "ner_silver_with_synthetic.jsonl"
COMB_CONLL = NER_DIR / "ner_silver_with_synthetic.conll"

NUM_SAMPLES = 2000  # sun'iy gaplar soni (xohlasang oshirasan)


# ==== Yordamchi funksiyalar ====

def gen_dates(n: int = 200) -> List[str]:
    """Turli formatlarda sana generatsiyasi."""
    res = []
    start = date(2020, 1, 1)
    for i in range(n):
        d = start + timedelta(days=i * 5)
        # 3 xil format
        res.append(f"{d.year} yil {d.day} {d.strftime('%B').lower()}")
        res.append(d.strftime("%d.%m.%Y"))
        res.append(d.strftime("%Y-%m-%d"))
    return list(set(res))


UZ_MONTHS = [
    "yanvar", "fevral", "mart", "aprel", "may", "iyun",
    "iyul", "avgust", "sentabr", "oktabr", "noyabr", "dekabr"
]

NAMES = [
    "Ali", "Vali", "Karim", "Dilshod", "Aziza", "Gulnoza",
    "Mahmud", "Oydin", "Sardor", "Nodira", "Javlon", "Sevara",
    "Jamshid", "Zafar", "Laylo", "Xurshid", "Madina", "Bobur",
]

CITIES = [
    "Toshkent", "Samarqand", "Buxoro", "Urganch", "Namangan",
    "Andijon", "Qo'qon", "Xiva", "Nukus", "Farg'ona",
]

POSITIONS = [
    "rektori", "professori", "dekani", "direktori",
    "vaziri", "hokimi", "raisi",
]

ORG_PATTERNS = [
    "{city} davlat universiteti",
    "{city} davlat pedagogika instituti",
    "{city} shahri hokimligi",
    "O'zbekiston Respublikasi Axborot texnologiyalari vazirligi",
    "O'zbekiston Respublikasi Fanlar akademiyasi",
]

PRODUCTS = [
    "Samsung Galaxy S24",
    "iPhone 15",
    "Redmi Note 13",
    "Galaxy Buds 3",
    "MacBook Air M2",
    "Dell XPS 15",
    "HP Spectre x360",
    "Asus ROG Strix",
]

# Template lar: matn + ishlatiladigan entity turlari
TEMPLATES = [
    # 1: Date + Org + Position + Person + Location
    {
        "template": "{DATE} kuni {ORG} {POSITION} {PERSON} {LOCATION}ga xizmat safariga jo'nab ketdi.",
        "slots": ["DATE", "ORG", "POSITION", "PERSON", "LOCATION"],
    },
    # 2: Org + Person + Position
    {
        "template": "{ORG} {POSITION} {PERSON} bugun matbuot anjumanida nutq so'zladi.",
        "slots": ["ORG", "POSITION", "PERSON"],
    },
    # 3: Person + Date + Org
    {
        "template": "{PERSON} {DATE} dan boshlab {ORG}da {POSITION} lavozimida ish boshladi.",
        "slots": ["PERSON", "DATE", "ORG", "POSITION"],
    },
    # 4: Product + Org + Location
    {
        "template": "{ORG} {LOCATION} shahrida yangi {PRODUCT} mahsulotini taqdimot qildi.",
        "slots": ["ORG", "LOCATION", "PRODUCT"],
    },
    # 5: Person + Location + Date
    {
        "template": "{PERSON} {LOCATION}ga {DATE} kuni ko'chib o'tdi.",
        "slots": ["PERSON", "LOCATION", "DATE"],
    },
    # 6: Org + Product + Date
    {
        "template": "{ORG} {DATE} kuni {PRODUCT} savdosini boshladi.",
        "slots": ["ORG", "DATE", "PRODUCT"],
    },
    # 7: Location + Org
    {
        "template": "{LOCATION} shahridagi {ORG} yangi o'quv yilini boshladi.",
        "slots": ["LOCATION", "ORG"],
    },
    # 8: Person + Product
    {
        "template": "{PERSON} yaqinda o'ziga {PRODUCT} sotib oldi va undan juda mamnun.",
        "slots": ["PERSON", "PRODUCT"],
    },
]


def make_org(city: str) -> str:
    pattern = random.choice(ORG_PATTERNS)
    return pattern.format(city=city)


def choose_date(dates: List[str]) -> str:
    return random.choice(dates)


def generate_one_sample(dates: List[str], idx: int) -> Dict:
    """
    Bitta sun'iy gap + entitylar ro'yxatini generatsiya qiladi.
    """
    tmpl = random.choice(TEMPLATES)
    text = tmpl["template"]

    city = random.choice(CITIES)
    person = random.choice(NAMES)
    org = make_org(city)
    position = random.choice(POSITIONS)
    product = random.choice(PRODUCTS)
    date_str = choose_date(dates)

    slot_values = {
        "PERSON": person,
        "LOCATION": city,
        "ORG": org,
        "POSITION": position,
        "PRODUCT": product,
        "DATE": date_str,
    }

    # Matnni yig'amiz va entity spanlarini hisoblaymiz
    entities: List[Dict] = []

    # string replace orqali joylashtiramiz, har bir slot uchun span topamiz
    text_filled = text

    for slot in tmpl["slots"]:
        val = slot_values[slot]
        # replace qilamiz (har biri matnda bir marta qatnashadi)
        # Lekin span topish uchun find ishlatamiz
        # (replace dan oldin ham, keyin ham bo'lishi mumkin)
        # Avval ensure qilamiz:
        placeholder = "{" + slot + "}"
        if placeholder in text_filled:
            text_filled = text_filled.replace(placeholder, val, 1)

    # Endi spanlarni topamiz
    for slot in tmpl["slots"]:
        val = slot_values[slot]
        start = text_filled.find(val)
        if start == -1:
            # qandaydir muammo bo'lsa, bu slotni tashlaymiz
            continue
        end = start + len(val)

        ent_type = {
            "PERSON": "Person",
            "LOCATION": "Location",
            "ORG": "Organization",
            "POSITION": "Position",
            "PRODUCT": "Product",
            "DATE": "Date",
        }[slot]

        entities.append({
            "text": val,
            "start": start,
            "end": end,
            "type": ent_type,
            "source": "synthetic",
        })

    # Token + BIO teglarga o'tkazamiz
    tokens, tags = text_to_bio(text_filled, entities)

    rec = {
        "id": idx,
        "text": text_filled,
        "tokens": tokens,
        "tags": tags,
        "entities": entities,
    }
    return rec


def text_to_bio(text: str, entities: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Matn + entity spanlar -> tokenlar va BIO teglar.
    """
    tokens: List[str] = []
    tags: List[str] = []

    if not text:
        return tokens, tags

    # entitylarni start bo'yicha tartiblaymiz
    ents_sorted = sorted(entities, key=lambda e: e["start"])

    words = text.split()
    offset = 0
    current_ent = None

    for w in words:
        raw = w
        start = text.find(raw, offset)
        if start == -1:
            start = offset
        end = start + len(raw)
        offset = end

        # ushbu token qaysi entity bilan kesishadi?
        overlapping = []
        for e in ents_sorted:
            if not (end <= e["start"] or e["end"] <= start):
                overlapping.append(e)

        if overlapping:
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


def save_jsonl(path: Path, records: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_conll(path: Path, records: List[Dict]):
    lines: List[str] = []
    for rec in records:
        sid = rec["id"]
        text = rec["text"]
        tokens = rec["tokens"]
        tags = rec["tags"]

        lines.append(f"# sent_id = {sid}")
        lines.append(f"# text = {text}")
        for tok, tag in zip(tokens, tags):
            lines.append(f"{tok}\t{tag}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    random.seed(42)

    print("NER_DIR:", NER_DIR)
    dates = gen_dates(300)
    print("Date misollari soni:", len(dates))

    synthetic_records: List[Dict] = []

    for i in range(NUM_SAMPLES):
        rec = generate_one_sample(dates, i)
        synthetic_records.append(rec)
        if (i + 1) % 200 == 0:
            print(f"{i + 1} ta sun'iy gap generatsiya qilindi...")

    print("Jami sun'iy gaplar:", len(synthetic_records))
    print("Sun'iy JSONL saqlanmoqda:", SYN_JSONL)
    save_jsonl(SYN_JSONL, synthetic_records)

    print("Sun'iy CoNLL saqlanmoqda:", SYN_CONLL)
    save_conll(SYN_CONLL, synthetic_records)

    # Agar original rules-based JSONL mavjud bo'lsa, birlashtiramiz
    combined_records: List[Dict] = []

    if ORIG_JSONL.exists():
        print("Original rules-based JSONL topildi, birlashtirayapmiz:", ORIG_JSONL)
        with ORIG_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                combined_records.append(rec)

    # Sun'iy yozuvlarni ham qo'shamiz (id larni siljitamiz)
    start_id = len(combined_records)
    for i, rec in enumerate(synthetic_records):
        rec2 = dict(rec)
        rec2["id"] = start_id + i
        combined_records.append(rec2)

    print("Birlashtirilgan JSONL saqlanmoqda:", COMB_JSONL)
    save_jsonl(COMB_JSONL, combined_records)

    print("Birlashtirilgan CoNLL saqlanmoqda:", COMB_CONLL)
    save_conll(COMB_CONLL, combined_records)

    print("Tayyor: synthetic + rules NER silver korpus yaratildi.")


if __name__ == "__main__":
    main()
