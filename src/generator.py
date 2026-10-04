"""Генерация синтетических данных через LLM."""
import json
import os
from typing import List
from openai import OpenAI
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.schemas import IntentSample, GenerationBatch


SYSTEM_PROMPT = """You are a data generation assistant.
Generate diverse, realistic samples for intent classification.
Return ONLY valid JSON matching this schema:
{
  "samples": [
    {"text": "...", "intent": "...", "language": "ru"|"en", "difficulty": "easy"|"medium"|"hard"}
  ]
}
Do NOT include markdown, explanations, or any text outside the JSON."""


class SyntheticGenerator:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.9):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.temperature = temperature

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _call_llm(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    def generate_batch(self, spec: dict) -> List[IntentSample]:
        """spec: {intent, language, n, examples}"""
        prompt = f"""Generate {spec['n']} samples for intent '{spec['intent']}' in language '{spec['language']}'.
Difficulty distribution: 30% easy, 50% medium, 20% hard.

Examples of similar real data:
{chr(10).join('- ' + e for e in spec.get('examples', []))}

Return JSON with 'samples' array."""
        raw = self._call_llm(prompt)
        try:
            batch = GenerationBatch.model_validate_json(raw)
            return batch.samples
        except ValidationError as e:
            print(f"Validation failed: {e}")
            return []

    def generate_dataset(self, specs: List[dict], output_path: str):
        all_samples = []
        for spec in specs:
            print(f"Generating {spec['n']} samples for {spec['intent']}...")
            samples = self.generate_batch(spec)
            all_samples.extend(samples)

        with open(output_path, "w", encoding="utf-8") as f:
            for s in all_samples:
                f.write(s.model_dump_json() + "\n")

        print(f"Saved {len(all_samples)} samples to {output_path}")
        return all_samples
