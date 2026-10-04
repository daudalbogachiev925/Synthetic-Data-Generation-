"""Утилиты: загрузка, сохранение, seed."""
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import List
from src.schemas import IntentSample


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: str) -> List[IntentSample]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            samples.append(IntentSample.model_validate_json(line))
    return samples


def save_jsonl(samples: List[IntentSample], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")
