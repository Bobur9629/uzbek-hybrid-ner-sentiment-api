# Uzbek Hybrid NER + Sentiment API

This repository contains a hybrid Uzbek-language NLP system that combines:

- **Rule-based + lexicon-based NER**
- **Transformer-based sentiment classifier (DistilBERT)**
- A simple **FastAPI** web service

### Project structure

- `src/`  – core Python modules (NER rules, ML NER wrapper, sentiment model, preprocessing, fusion, FastAPI server)
- `data/lexicons/` – dictionaries for persons, locations, organizations, positions, products, emojis, normalization pairs
- `data/processed/` – cleaned sentiment dataset and silver NER corpora
- `models/` – local trained models (not included in this repo, used only on the author's machine)

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
