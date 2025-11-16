# src/debug_ner_ml_only.py

from pprint import pprint

from .ner_ml import NERML
from .preprocessing import preprocess_text


if __name__ == "__main__":
    model = NERML()  # configdagi NER_BASE_MODEL bo'yicha yuklaydi

    text = (
        "Urganch davlat universiteti rektori Ali 2024 yil Toshkentga safar qildi. "
        "U yerda Samsung Galaxy S24 va iPhone 15 mahsulotlari taqdimot qilindi."
    )

    print("Original:", text)
    prep = preprocess_text(text)
    print("Preprocessed:", prep)

    ents_ml = model(prep)
    print("\n=== ML-NER natijasi (source='ml') ===")
    pprint(ents_ml)
