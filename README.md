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

Author: Bobur Saidov
Contact: saidovboburbek9629@gmail.com
