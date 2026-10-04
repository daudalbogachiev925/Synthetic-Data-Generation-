"""Тесты валидатора."""
from src.schemas import IntentSample
from src.validator import DataValidator


def test_class_balance():
    samples = [
        IntentSample(text=f"a{i}", intent="greet", language="ru") for i in range(5)
    ] + [
        IntentSample(text=f"b{i}", intent="order", language="ru") for i in range(5)
    ]
    v = DataValidator(samples)
    balance = v.class_balance()
    assert balance["greet"] == 0.5
    assert balance["order"] == 0.5


def test_duplicate_rate():
    samples = [
        IntentSample(text="hello", intent="greet", language="en"),
        IntentSample(text="hello", intent="greet", language="en"),
        IntentSample(text="hi", intent="greet", language="en"),
    ]
    v = DataValidator(samples)
    assert v.duplicate_rate() > 0


def test_diversity_score():
    samples = [
        IntentSample(text="hello world", intent="greet", language="en"),
        IntentSample(text="completely different text", intent="order", language="en"),
    ]
    v = DataValidator(samples)
    assert v.diversity_score() > 0
