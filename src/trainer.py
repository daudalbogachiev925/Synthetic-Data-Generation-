"""Обучение downstream-модели на синтетике / реальных / миксе."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from typing import List, Tuple
from src.schemas import IntentSample


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])


def train_and_eval(
    train_samples: List[IntentSample],
    test_samples: List[IntentSample],
) -> dict:
    X_train = [s.text for s in train_samples]
    y_train = [s.intent for s in train_samples]
    X_test = [s.text for s in test_samples]
    y_test = [s.intent for s in test_samples]

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1_macro": round(f1_score(y_test, preds, average="macro"), 4),
        "report": classification_report(y_test, preds, zero_division=0),
    }
