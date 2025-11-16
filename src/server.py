# src/server.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from src.hybrid_model import HybridUzbekModel, HybridUzbekModelV01

app = FastAPI(
    title="Uzbek Hybrid NER + Sentiment API",
    version="0.1.0",
)

# ---- Pydantic modellari ----

class TextIn(BaseModel):
    text: str


class EntityOut(BaseModel):
    text: str
    start: int
    end: int
    type: str
    source: str


class AnalysisOut(BaseModel):
    text_original: str
    text_preprocessed: str

    sentiment: str
    ml_label: Optional[str] = None
    ml_score: Optional[float] = None
    lex_label: Optional[str] = None
    lex_score: Optional[float] = None

    entities: List[EntityOut]
    source: str


print("[server] HybridUzbekModelV01 (rules-only) yuklanmoqda...")
model_rules = HybridUzbekModelV01()

print("[server] HybridUzbekModel (full ML+rules) yuklanmoqda...")
model_full = HybridUzbekModel(use_ml_ner=True)

print("[server] Barcha modelllar yuklandi.")


@app.get("/")
def root():
    return {"message": "Uzbek Hybrid NER + Sentiment API ishlayapti"}


@app.post("/analyze/rules", response_model=AnalysisOut)
def analyze_rules(req: TextIn) -> AnalysisOut:
    res = model_rules.analyze(req.text)
    return AnalysisOut(**res)


@app.post("/analyze/full", response_model=AnalysisOut)
def analyze_full(req: TextIn) -> AnalysisOut:
    res = model_full.analyze(req.text)
    return AnalysisOut(**res)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=False)
