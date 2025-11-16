# src/ner_ml.py

from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from .config import NER_BASE_MODEL


def _words_with_char_spans(text: str):
    """
    Matnni .split() orqali tokenlab,
    har bir token uchun (word, start, end) qaytaradi.
    """
    words = text.split()
    spans = []
    offset = 0
    for w in words:
        start = text.find(w, offset)
        if start == -1:
            start = offset
        end = start + len(w)
        spans.append((w, start, end))
        offset = end
    return words, spans


class NERML:
    """
    Transformer-based NER (token classification).

    - Model va tokenizer NER_BASE_MODEL dan yuklanadi (config.py).
    - BIO teglar (O, B-Location, I-Product, ...) asosida
      char-level spanlar qaytaradi.
    """

    def __init__(self, model_path: Optional[str] = None, max_length: int = 128):
        if model_path is None:
            model_path = NER_BASE_MODEL

        self.model_path = model_path
        self.max_length = max_length

        print(f"[NERML] Modelni yuklayapman: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        # id2label config ichida saqlangan
        self.id2label = self.model.config.id2label

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _predict_tags(self, text: str) -> List[str]:
        """
        Matn -> word-level BIO teglar ro'yxati.
        (subtokenlar uchun faqat birinchi subtoken tegini olamiz)
        """
        words, _ = _words_with_char_spans(text)

        if not words:
            return []

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        word_ids = enc.word_ids(batch_index=0)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            logits = outputs.logits  # (1, seq_len, num_labels)
            preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Har bir subtoken uchun label id → word-level'ga agregatsiya
        tags_sub: List[Optional[str]] = []
        prev_word_idx = None
        for idx, w_id in enumerate(word_ids):
            if w_id is None:
                tags_sub.append(None)
                continue

            if w_id != prev_word_idx:
                label_id = int(preds[idx])
                tags_sub.append(self.id2label[label_id])
            else:
                tags_sub.append(None)

            prev_word_idx = w_id

        # word-level ketma-ketlikni olamiz
        word_level_tags: List[str] = []
        seen = set()
        for idx, w_id in enumerate(word_ids):
            if w_id is None:
                continue
            if w_id in seen:
                continue
            seen.add(w_id)
            tag = tags_sub[idx]
            if tag is None:
                tag = "O"
            word_level_tags.append(tag)

        assert len(word_level_tags) == len(words), "Token/teg uzunligi mos emas!"
        return word_level_tags

    def _bio_to_spans(self, text: str, tags: List[str]) -> List[Dict]:
        """
        Matn + word-level BIO teglar -> entity spanlar (char start/end bilan).
        """
        words, spans = _words_with_char_spans(text)
        entities: List[Dict] = []

        cur_type = None
        cur_start_char = None
        cur_end_char = None

        for (word, (w, start, end)), tag in zip(zip(words, spans), tags):
            if tag == "O" or tag is None:
                if cur_type is not None:
                    entities.append({
                        "text": text[cur_start_char:cur_end_char],
                        "start": cur_start_char,
                        "end": cur_end_char,
                        "type": cur_type,
                        "source": "ml",
                    })
                    cur_type = None
                    cur_start_char = None
                    cur_end_char = None
                continue

            if tag.startswith("B-"):
                # Avvalgi entityni yopamiz (agar bo'lsa)
                if cur_type is not None:
                    entities.append({
                        "text": text[cur_start_char:cur_end_char],
                        "start": cur_start_char,
                        "end": cur_end_char,
                        "type": cur_type,
                        "source": "ml",
                    })

                ent_type = tag[2:]
                cur_type = ent_type
                cur_start_char = start
                cur_end_char = end

            elif tag.startswith("I-") and cur_type is not None:
                ent_type = tag[2:]
                if ent_type == cur_type:
                    cur_end_char = end
                else:
                    # noto'g'ri I- turi: avvalgisini yopamiz, yangisini B- deb olamiz
                    entities.append({
                        "text": text[cur_start_char:cur_end_char],
                        "start": cur_start_char,
                        "end": cur_end_char,
                        "type": cur_type,
                        "source": "ml",
                    })
                    cur_type = ent_type
                    cur_start_char = start
                    cur_end_char = end

        # Oxirgi entityni ham qo'shamiz
        if cur_type is not None:
            entities.append({
                "text": text[cur_start_char:cur_end_char],
                "start": cur_start_char,
                "end": cur_end_char,
                "type": cur_type,
                "source": "ml",
            })

        entities = sorted(entities, key=lambda x: x["start"])
        return entities

    def __call__(self, text: str) -> List[Dict]:
        """
        Asosiy chaqiruv:
        text -> [ {text, start, end, type, source="ml"}, ... ]
        """
        text = str(text)
        if not text.strip():
            return []

        try:
            tags = self._predict_tags(text)
        except Exception as e:
            print(f"[NERML] Xato, bo'sh ro'yxat qaytaryapman: {e}")
            return []

        ents = self._bio_to_spans(text, tags)
        return ents
