"Тесты генератора."
import pytest
from src.schemas import IntentSample, GenerationBatch
from pydantic import ValidationError


def test_intent_sample_valid():
    s = IntentSample(text="привет", intent="greet", language="ru")
    assert s.text == "привет"


def test_intent_sample_invalid_intent():
    with pytest.raises(ValidationError):
        IntentSample(text="hi", intent="unknown", language="ru")


def test_intent_sample_blank_text():
    with pytest.raises(ValidationError):
        IntentSample(text="   ", intent="greet", language="ru")


def test_batch_validation():
    batch = GenerationBatch(samples=[
        {"text": "hello", "intent": "greet", "language": "en"},
        {"text": "bye", "intent": "goodbye", "language": "en"},
    ])
    assert len(batch.samples) == 2
