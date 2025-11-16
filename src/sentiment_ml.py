# src/sentiment_ml.py

from typing import Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class SentimentML:
    def __init__(self, model_name_or_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )
        self.model.to(self.device)
        self.model.eval()

        # label mappingni model config dan olamiz
        self.id2label = self.model.config.id2label
        # himoya uchun lower-case
        self.id2label = {k: v.lower() for k, v in self.id2label.items()}

    def __call__(self, text: str) -> Tuple[str, float]:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)
            probs = torch.softmax(outputs.logits, dim=-1)
            score, pred_id = torch.max(probs, dim=-1)

        label = self.id2label[int(pred_id)]
        return label, float(score)
