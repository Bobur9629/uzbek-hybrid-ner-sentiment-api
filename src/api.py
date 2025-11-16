# src/api.py

from typing import Dict, Any

from .hybrid_model import HybridUzbekModel, HybridUzbekModelV01


# Bitta global instance – modelni har chaqirganda qayta yuklamaslik uchun
_rules_model = HybridUzbekModelV01()

# To'liq gibrid model (ML-sentiment + rules-NER + ixtiyoriy ML-NER)
_full_model = HybridUzbekModel(
    ner_model_path="L:/Hybrid model/models/ner/uz_ner_silver_xlmr/best",
    sent_model_path=None,   # SENT_BASE_MODEL dan oladi
    use_ml_ner=True,
    debug=False,
)


def analyze_rules_only(text: str) -> Dict[str, Any]:
    """
    Faqat rules-NER + leksikon-sentiment varianti.
    """
    return _rules_model.analyze(text)


def analyze_full(text: str) -> Dict[str, Any]:
    """
    To'liq gibrid variant:
    - NER: rules (+ ML-NER, agar sozlangan bo'lsa)
    - Sentiment: ML + leksikon + fusion
    """
    return _full_model.analyze(text)


if __name__ == "__main__":
    from pprint import pprint

    s = "2024 yil 5 mart kuni Urganch davlat universiteti rektori Ali Toshkentga safar qildi."
    print("=== API: rules-only ===")
    pprint(analyze_rules_only(s))

    print("\n=== API: full hybrid ===")
    pprint(analyze_full(s))
