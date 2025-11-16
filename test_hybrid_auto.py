# src/test_hybrid_auto.py

from pprint import pprint
from .hybrid_model import HybridUzbekModel


def find_entity(entities, text_substr, ent_type):
    """
    entities ro'yxatida substring + type bo'yicha qidiradi.
    """
    for e in entities:
        if text_substr in e["text"] and e["type"] == ent_type:
            return True
    return False


def main():
    model = HybridUzbekModel()

    tests = [
        {
            "name": "Positive with emoji",
            "text": "Bu film menga juda yoqdi, aktyorlar zo'r o'ynagan 😊",
            "expect_sent": "positive",
        },
        {
            "name": "Negative with emoji",
            "text": "Bu xizmat juda yomon, umuman yoqmadi 😡",
            "expect_sent": "negative",
        },
        {
            "name": "Neutral factual sentence",
            "text": "2024 yil 5 mart kuni Urganch davlat universiteti rektori Ali Toshkentga safar qildi.",
            "expect_sent": "neutral",
            "expect_entities": [
                ("2024 yil", "Date"),
                ("5 mart", "Date"),
                ("Urganch davlat universiteti", "Organization"),
                ("Ali", "Person"),
                ("Toshkentga", "Location"),
            ],
        },
    ]

    for t in tests:
        print(f"\n=== Test: {t['name']} ===")
        res = model.analyze(t["text"])
        pprint(res)

        # Sentiment check
        got_sent = res["sentiment"]
        if got_sent == t["expect_sent"]:
            print(f"[OK] Sentiment: {got_sent}")
        else:
            print(f"[FAIL] Sentiment: {got_sent} (expected: {t['expect_sent']})")

        # Entity check (if provided)
        if "expect_entities" in t:
            for substr, etype in t["expect_entities"]:
                ok = find_entity(res["entities"], substr, etype)
                if ok:
                    print(f"[OK] Entity: '{substr}' as {etype}")
                else:
                    print(f"[FAIL] Entity: '{substr}' as {etype} topilmadi")

    print("\nTestlar tugadi.")


if __name__ == "__main__":
    main()
