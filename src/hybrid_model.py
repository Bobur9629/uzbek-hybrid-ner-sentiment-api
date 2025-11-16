# src/hybrid_model.py

from typing import Dict, Any, Optional

from .preprocessing import preprocess_text
from .ner_ml import NERML
from .ner_rules import NERRules
from .sentiment_ml import SentimentML
from .sentiment_lex import SentimentLexicon
from .fusion import merge_entities, fuse_sentiment
from .config import SENT_BASE_MODEL  # sentiment modeli yo‘li (distilbert)


class HybridUzbekModelV01:
    """
    1-versiya (faqat rules + leksikon sentiment):
    - NER: faqat NERRules (Person, Position, Organization, Product, Location, Date)
    - Sentiment: faqat SentimentLexicon (emoji + so'z lug'ati bo'lsa)
    ML-modelsiz tez test qilish uchun.
    """

    def __init__(self):
        self.ner_rules = NERRules()
        self.sent_lex = SentimentLexicon()

    def analyze(self, text: str) -> Dict[str, Any]:
        text_prep = preprocess_text(text)

        # Faqat rules NER
        ents_rules = self.ner_rules(text_prep)

        # Faqat leksikon sentiment
        lex_label, lex_score = self.sent_lex(text_prep)

        return {
            "text_original": text,
            "text_preprocessed": text_prep,
            "sentiment": lex_label,
            "sentiment_score": lex_score,
            "entities": ents_rules,
            "source": "rules_only",
        }


class HybridUzbekModel:
    """
    To'liq gibrid model (ML-sentiment + rules-NER + leksikon).

    - NER:
        * NERRules (lug'at + regex)
        * ixtiyoriy: NERML (transformer) — agar ner_model_path berilgan bo'lsa
    - Sentiment:
        * SentimentML (transformer)
        * SentimentLexicon (emoji + so'z lug'ati)
        va natijalar fuse_sentiment orqali birlashtiriladi.
    """

    def __init__(
        self,
        ner_model_path: Optional[str] = None,
        sent_model_path: Optional[str] = None,
        use_ml_ner: bool = True,
        debug: bool = False,
    ):
        """
        :param ner_model_path: NERML modeli papkasi (best/), None bo'lsa NERML o'chiriladi
        :param sent_model_path: SentimentML modeli papkasi, None bo'lsa SENT_BASE_MODEL ishlatiladi
        :param use_ml_ner: True bo'lsa, ML-NER yoqiladi, False bo'lsa faqat rules-NER ishlaydi
        :param debug: True bo'lsa ba'zi intermediate natijalar konsolga chiqariladi
        """
        self.debug = debug

        # NER qismini sozlash
        self.ner_rules = NERRules()
        self.use_ml_ner = use_ml_ner and (ner_model_path is not None)
        self.ner_ml = None
        if self.use_ml_ner:
            self.ner_ml = NERML(ner_model_path)

        # Sentiment qismini sozlash
        if sent_model_path is None:
            sent_model_path = SENT_BASE_MODEL
        self.sent_ml = SentimentML(sent_model_path)
        self.sent_lex = SentimentLexicon()

    def analyze(self, text: str) -> Dict[str, Any]:
        text_prep = preprocess_text(text)

        # --- NER: rules + (ixtiyoriy) ML ---
        ents_rules = self.ner_rules(text_prep)
        ents_ml = []

        if self.use_ml_ner and self.ner_ml is not None:
            ents_ml = self.ner_ml.predict(text_prep)

        if self.debug:
            print("\n[DEBUG] ents_rules:")
            print(ents_rules)
            if ents_ml:
                print("[DEBUG] ents_ml:")
                print(ents_ml)

        if ents_ml:
            entities = merge_entities(ents_ml, ents_rules)
        else:
            entities = ents_rules

        if self.debug:
            print("[DEBUG] merged entities:")
            print(entities)

        # --- Sentiment: ML + lexicon ---
        ml_label, ml_score = self.sent_ml(text_prep)
        lex_label, lex_score = self.sent_lex(text_prep)
        sentiment = fuse_sentiment(ml_label, ml_score, lex_label, lex_score)

        return {
            "text_original": text,
            "text_preprocessed": text_prep,
            "sentiment": sentiment,
            "ml_label": ml_label,
            "ml_score": ml_score,
            "lex_label": lex_label,
            "lex_score": lex_score,
            "entities": entities,
            "source": "hybrid_mlSent+rulesNER" if self.use_ml_ner else "hybrid_rulesNER_only",
        }


if __name__ == "__main__":
    # Oddiy self-test
    from pprint import pprint

    sample = (
        "Urganch davlat universiteti rektori Ali 2024 yil Toshkentga safar qildi. "
        "U yerda Samsung Galaxy S24 va iPhone 15 mahsulotlari taqdimot qilindi."
    )

    print("=== HybridUzbekModelV01 (rules only) testi ===")
    model_rules = HybridUzbekModelV01()
    pprint(model_rules.analyze(sample))

    print("\n=== HybridUzbekModel (ML-sentiment + rules-NER + lexicon) testi ===")
    # Bu yerda NERML vaqtincha o‘chirilgan – faqat rules NER ishlaydi,
    # sent_model esa SENT_BASE_MODEL dan olinadi
    model_full = HybridUzbekModel(
        ner_model_path="L:/Hybrid model/models/ner/uz_ner_silver_xlmr/best",
        sent_model_path=None,
        use_ml_ner=True,
        debug=False,  # keraksiz print'lar bo'lmasin
    )
    pprint(model_full.analyze(sample))
