"""Pydantic schemas для валидации сгенерированных данных."""
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal


class IntentSample(BaseModel):
    """Один пример для intent classification."""
    text: str = Field(..., min_length=3, max_length=200)
    intent: Literal["greet", "order", "cancel", "ask_price", "goodbye", "other"]
    language: Literal["ru", "en"] = "ru"
    difficulty: Literal["easy", "medium", "hard"] = "medium"

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text cannot be blank")
        return v.strip()


class QAExample(BaseModel):
    """Пример Q&A для SFT."""
    instruction: str = Field(..., min_length=5)
    response: str = Field(..., min_length=5)
    category: Literal["technical", "general", "math", "code"]

    @field_validator("response")
    @classmethod
    def response_has_content(cls, v: str) -> str:
        if len(v.split()) < 3:
            raise ValueError("response too short")
        return v


class GenerationBatch(BaseModel):
    """Батч сгенерированных примеров — то, что возвращает LLM."""
    samples: List[IntentSample]
