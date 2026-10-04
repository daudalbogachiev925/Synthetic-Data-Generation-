"""Валидация и анализ сгенерированных данных."""
import json
from collections import Counter
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.schemas import IntentSample


class DataValidator:
    def __init__(self, samples: List[IntentSample]):
        self.samples = samples

    def class_balance(self) -> dict:
        """Проверка баланса классов."""
        counts = Counter(s.intent for s in self.samples)
        total = sum(counts.values())
        return {k: round(v / total, 3) for k, v in counts.items()}

    def length_stats(self) -> dict:
        lengths = [len(s.text.split()) for s in self.samples]
        return {
            "mean": round(np.mean(lengths), 2),
            "std": round(np.std(lengths), 2),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
        }

    def diversity_score(self) -> float:
        """Средняя cosine distance между случайными парами — выше = разнообразнее."""
        texts = [s.text for s in self.samples]
        if len(texts) < 2:
            return 0.0
        vec = TfidfVectorizer(max_features=500).fit_transform(texts)
        sim = cosine_similarity(vec)
        np.fill_diagonal(sim, 0)
        return round(1 - sim.mean(), 3)

    def duplicate_rate(self) -> float:
        texts = [s.text.lower().strip() for s in self.samples]
        return round(1 - len(set(texts)) / len(texts), 3)

    def report(self) -> dict:
        return {
            "n_samples": len(self.samples),
            "class_balance": self.class_balance(),
            "length_stats": self.length_stats(),
            "diversity_score": self.diversity_score(),
            "duplicate_rate": self.duplicate_rate(),
        }
