# training/train_ner_silver.py

from pathlib import Path
from typing import List, Dict, Tuple

import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)

from src.config import BASE_DIR, PROC_DIR

# -----------------------
# Sozlamalar
# -----------------------

NER_DATA_DIR = PROC_DIR / "ner_silver"
NER_JSONL_PATH = NER_DATA_DIR / "ner_silver_with_synthetic.jsonl"

# Agar xotira yetmasa, buni "distilbert-base-multilingual-cased" ga almashtirishing mumkin
BASE_MODEL_NAME = "distilbert-base-multilingual-cased"

OUT_DIR = BASE_DIR / "models" / "ner" / "uz_ner_silver_xlmr"
BEST_DIR = OUT_DIR / "best"
BEST_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 128
BATCH_SIZE_TRAIN = 4
BATCH_SIZE_VAL = 8
NUM_EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
RANDOM_SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# JSONL o'qish va label lug'at
# -----------------------

def load_ner_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
    return records


def build_label_vocab(records: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    tag_set = set()
    for rec in records:
        for t in rec["tags"]:
            tag_set.add(t)

    # Har doim 'O' ni birinchi qilish yaxshi, qolganlarini tartiblab chiqamiz
    tags = sorted(list(tag_set))
    if "O" in tags:
        tags.remove("O")
        tags = ["O"] + tags

    label2id = {tag: i for i, tag in enumerate(tags)}
    id2label = {i: tag for tag, i in label2id.items()}
    return label2id, id2label


# -----------------------
# Dataset klassi
# -----------------------

class NerDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        tokens: List[str] = rec["tokens"]
        tags: List[str] = rec["tags"]

        # HuggingFace tokenizatsiyasi: so'zlar ro'yxatini beramiz
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # word_ids: har bir subtoken qaysi original token(word)ga tegishli
        word_ids = enc.word_ids(batch_index=0)

        labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)  # pad/CLS/SEP uchun
            else:
                tag = tags[word_idx]
                # Birinchi subtoken uchun to'liq tag, qolganlariga -100
                if word_idx != previous_word_idx:
                    labels.append(self.label2id[tag])
                else:
                    labels.append(-100)
            previous_word_idx = word_idx

        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(labels, dtype=torch.long)

        return enc


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
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader, id2label, device):
    model.eval()
    all_true = []
    all_pred = []

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

            # Flatten qilamiz, -100 larni tashlaymiz
            for true_seq, pred_seq in zip(labels, preds):
                for t, p in zip(true_seq.cpu().numpy(), pred_seq.cpu().numpy()):
                    if t == -100:
                        continue
                    all_true.append(t)
                    all_pred.append(p)

    # Token-level F1 (O ham ichida)
    f1 = f1_score(all_true, all_pred, average="macro")
    print("Classification report (token-level):")
    print(classification_report(
        all_true,
        all_pred,
        target_names=[id2label[i] for i in sorted(set(all_true))],
        digits=4,
    ))
    return f1


# -----------------------
# Asosiy main()
# -----------------------

def main():
    set_seed(RANDOM_SEED)

    print("NER JSONL fayl:", NER_JSONL_PATH)
    records = load_ner_jsonl(NER_JSONL_PATH)
    print("Jami yozuvlar:", len(records))

    label2id, id2label = build_label_vocab(records)
    print("Yorliqlar:", label2id)

    # Train/val bo'linishi (oddiy random split)
    random.shuffle(records)
    val_ratio = 0.1
    split_idx = int(len(records) * (1 - val_ratio))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    print("Train size:", len(train_records))
    print("Val size:", len(val_records))

    print("Tokenizer va modelni yuklayapman:", BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model.to(device)

    train_dataset = NerDataset(train_records, tokenizer, label2id, max_length=MAX_LEN)
    val_dataset = NerDataset(val_records, tokenizer, label2id, max_length=MAX_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
    )

    num_training_steps = NUM_EPOCHS * len(train_loader)
    num_warmup_steps = int(WARMUP_RATIO * num_training_steps)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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

        val_f1 = evaluate(model, val_loader, id2label, device)
        print(f"Val F1 (macro, token-level): {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            print("Yangi eng yaxshi model! Saqlayapman...")
            model.save_pretrained(str(BEST_DIR))
            tokenizer.save_pretrained(str(BEST_DIR))

    print("\nTrening tugadi.")
    print("Eng yaxshi F1:", best_f1)
    print("Model papkasi:", BEST_DIR)


if __name__ == "__main__":
    main()
