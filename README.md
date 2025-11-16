# Uzbek Hybrid NER + Sentiment API

Hybrid rule-based + transformer NER and sentiment analysis model for the Uzbek language with a simple FastAPI web service.

This repository contains:
- **Rule-based NER** (lexicon + patterns)
- **Transformer-based NER** (XLM-RoBERTa fine-tuned on silver data)
- **Transformer-based sentiment model** (DistilBERT for Uzbek)
- **Lexicons and datasets** used for training
- **FastAPI server** to expose the models as a REST API

## Project structure

```text
.
├── src/                    # Python package with all runtime code
│   ├── server.py           # FastAPI app (entrypoint)
│   ├── hybrid_model.py     # High-level hybrid model
│   ├── ner_rules.py        # Rule-based NER
│   ├── ner_ml.py           # ML NER wrapper
│   ├── sentiment_ml.py     # ML sentiment wrapper
│   ├── sentiment_lex.py    # Lexicon-based sentiment
│   ├── preprocessing.py    # Text preprocessing utilities
│   └── fusion.py           # Fusion of rules + ML
│
├── data/
│   ├── lexicons/           # Person, location, product, emoji lexicons, etc.
│   ├── processed/          # Cleaned training corpora (NER & sentiment)
│   └── raw/                # Original raw Excel datasets
│
├── training/               # Scripts for training models
│   ├── train_ner_silver.py
│   ├── train_sentiment.py
│   ├── generate_synthetic_ner_data.py
│   └── prepare_ner_silver.py
└── models/                 # Trained model weights (NER & sentiment)


### Installation

```bash
git clone https://github.com/Bobur9629/uzbek-hybrid-ner-sentiment-api.git
cd uzbek-hybrid-ner-sentiment-api

# (optional) create virtual env / conda env
pip install -r requirements.txt

running the API - python -m src.server
By default the API will be available at: http://127.0.0.1:8000
You can open interactive Swagger docs: http://127.0.0.1:8000/docs

Example request (full hybrid model)
curl -X POST "http://127.0.0.1:8000/analyze/full" \
  -H "Content-Type: application/json" \
  -d '{"text": "Urganch davlat universiteti rektori Ali 2024 yil Toshkentga safar qildi."}'

Endpoints

POST /analyze/rules – rule-based NER + lexicon sentiment only
POST /analyze/full – ML sentiment + rule-based NER + fusion

Example response (shortened):

{
  "text_original": "Urganch davlat universiteti rektori Ali 2024 yil Toshkentga safar qildi.",
  "sentiment": "positive",
  "ml_label": "positive",
  "entities": [
    {"text": "Urganch", "type": "Location", "source": "rules"},
    {"text": "Urganch davlat universiteti", "type": "Organization", "source": "ml+rules"},
    {"text": "Ali", "type": "Person", "source": "ml+rules"},
    {"text": "2024 yil", "type": "Date", "source": "ml+rules"},
    {"text": "Toshkentga", "type": "Location", "source": "rules"}
  ],
  "source": "hybrid_mlSent+rulesNER"
}

Training scripts

All training scripts are located in the training/ directory:

train_ner_silver.py – train NER model on silver data

train_sentiment.py – train DistilBERT sentiment model

generate_synthetic_ner_data.py – create synthetic NER examples

prepare_ner_silver.py – preprocessing and silver-label pipeline

These scripts are provided for reproducibility of the experiments described in the paper.


Key: README bo‘ldi – retsenzent ham, boshqa dasturchi ham tushunadi.

---

## 2️⃣ requirements.txt ni tayyorlash

Loyihani qaysi env’da o‘qitgan bo‘lsangiz, o‘sha **Anaconda Prompt**da:

```bash
cd "L:\Hybrid model"   # yoki clone qilingan papkangiz
pip freeze > requirements.txt

Author: Bobur Saidov Rashidovich tel:+998942326227
Contact: saidovboburbek9629@gmail.com
