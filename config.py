# src/config.py

from pathlib import Path
from typing import List

# --- Loyihaning asosiy yo'li ---

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data papkalar ---

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
LEX_DIR = DATA_DIR / "lexicons"

# Eski kodlar bilan moslik uchun
LEXICON_DIR = LEX_DIR

# --- Leksikon fayllari yo'llari ---

EMOJI_LEXICON_PATH = LEX_DIR / "emoji_lexicon.csv"
EMOJI_ASCII_PATH = LEX_DIR / "emoji_ascii.csv"
NORMALIZATION_PAIRS_PATH = LEX_DIR / "uz_normalization_pairs.csv"

PERSON_DICT_PATH = LEX_DIR / "person_dict.txt"
LOC_DICT_PATH = LEX_DIR / "loc_dict.txt"
POSITION_DICT_PATH = LEX_DIR / "position_dict.txt"
ORG_DICT_PATH = LEX_DIR / "org_dict.txt"
PRODUCT_DICT_PATH = LEX_DIR / "product_dict.txt"

# Sentiment dataset
SENTIMENT_DATA_PATH = PROC_DIR / "sentiment_uz_9970_clean.csv"

# --- Model yo'llari ---

# NER: silver + synthetic bo'yicha trenirovka qilingan model
NER_BASE_MODEL = str(
    BASE_DIR / "models" / "ner" / "uz_ner_silver_xlmr" / "best"
)

# Sentiment: DistilBERT (train_sentiment.py bilan o'qitilgan)
SENT_BASE_MODEL = str(
    BASE_DIR / "models" / "sentiment" / "uz_distilbert_sentiment" / "best"
)

# Moslik uchun aliaslar
NER_MODEL_NAME = NER_BASE_MODEL
SENTIMENT_MODEL_NAME = SENT_BASE_MODEL

# --- Label to'plamlari ---

ENTITY_TYPES: List[str] = [
    "Person",
    "Position",
    "Organization",
    "Date",
    "Location",
    "Product",
]

SENTIMENT_LABELS: List[str] = [
    "positive",
    "negative",
    "neutral",
]
