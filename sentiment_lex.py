# src/sentiment_lex.py

import csv
from pathlib import Path
from .config import LEXICON_DIR

class SentimentLexicon:
    def __init__(self):
        self.word_polarity = self._load_word_lexicon(LEXICON_DIR / "sentiment_lexicon.csv")
        self.emoji_polarity = self._load_emoji_lexicon(LEXICON_DIR / "emoji_lexicon.csv")

    @staticmethod
    def _load_word_lexicon(path: Path):
        d = {}
        if not path.exists():
            return d
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            # ustun nomlari sendagi faylga qarab: word;polarity;weight
            for row in reader:
                w = row["word"].lower().strip()
                wgt = float(row.get("weight", row.get("polarity", 0)))
                d[w] = wgt
        return d

    @staticmethod
    def _load_emoji_lexicon(path: Path):
        d = {}
        if not path.exists():
            return d
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            # c;polarity
            for row in reader:
                emo = row["emoji"]
                val = float(row.get("weight", row.get("polarity", 0)))
                d[emo] = val
        return d

    def __call__(self, text: str):
        score = 0.0

        # emoji lar
        for ch in text:
            if ch in self.emoji_polarity:
                score += self.emoji_polarity[ch]

        # so'zlar
        for w in text.lower().split():
            if w in self.word_polarity:
                score += self.word_polarity[w]

        # score ni labelga o'tkazish (thresholdlar keyin kalibrovka qilinadi)
        if score > 0.5:
            label = "positive"
        elif score < -0.5:
            label = "negative"
        else:
            label = "neutral"
        return label, score
