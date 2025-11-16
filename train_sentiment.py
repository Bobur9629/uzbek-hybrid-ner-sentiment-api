import os
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# -----------------------
# Yo'llar va sozlamalar
# -----------------------

BASE_DIR = Path(__file__).resolve().parent.parent   # L:/Hybrid model
DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "sentiment"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = PROC_DIR / "sentiment_uz_9970_clean.csv"

# XLM-R o'rniga yengilroq ko'p tilli model
BASE_MODEL_NAME = "distilbert-base-multilingual-cased"

OUTPUT_DIR = MODELS_DIR / "uz_distilbert_sentiment"
BEST_DIR = OUTPUT_DIR / "best"
BEST_DIR.mkdir(parents=True, exist_ok=True)

# Kichik GPU uchun parametrlar
MAX_LEN = 64          # 64 token yetarli
TRAIN_BS = 8          # distilbert yengil, 8/16 normal bo'ladi
VAL_BS = 16
NUM_EPOCHS = 3

LABEL2ID: Dict[str, int] = {
    "positive": 0,
    "negative": 1,
    "neutral": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# -----------------------
# Dataset klassi
# -----------------------

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# -----------------------
# Train / eval funksiyalari
# -----------------------

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    return acc, f1_macro


# -----------------------
# Asosiy trening jarayoni
# -----------------------

def main():
    print("CSV ni o‘qiyapman:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    assert "text" in df.columns and "sentiment" in df.columns, "CSV ustunlari mos emas!"

    def map_label(lbl: str) -> int:
        lbl = str(lbl).strip().lower()
        if lbl not in LABEL2ID:
            return LABEL2ID["neutral"]
        return LABEL2ID[lbl]

    df["label_id"] = df["sentiment"].apply(map_label)

    train_df, val_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df["label_id"],
    )

    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Label taqsimoti (train):")
    print(train_df["sentiment"].value_counts())

    print("Tokenizer va modelni yuklayapman:", BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model.to(device)

    train_dataset = SentimentDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LEN,
    )

    val_dataset = SentimentDataset(
        texts=val_df["text"].tolist(),
        labels=val_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LEN,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BS,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BS,
        shuffle=False,
    )

    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    num_training_steps = NUM_EPOCHS * len(train_loader)
    num_warmup_steps = int(0.06 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_f1 = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")

        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Val accuracy: {val_acc:.4f}")
        print(f"Val F1-macro: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            print("Yangi eng yaxshi model! Saqlayapman...")
            model.save_pretrained(str(BEST_DIR))
            tokenizer.save_pretrained(str(BEST_DIR))

    print("\nTrening tugadi.")
    print("Eng yaxshi F1-macro:", best_f1)
    print("Model papkasi:", BEST_DIR)


if __name__ == "__main__":
    main()
