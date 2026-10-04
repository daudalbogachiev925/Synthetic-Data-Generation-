# Synthetic Data Factory

Generate high-quality synthetic datasets with LLMs, validate them, and train downstream models.

## Problem

Real labeled data is expensive. LLMs can generate synthetic data, but quality control is hard:
- Duplicates, hallucinations, format errors
- Distribution mismatch with real data
- No systematic validation

## Solution

End-to-end pipeline:
1. **Generate** — LLM produces structured samples (Pydantic-validated)
2. **Validate** — schema + semantic checks + diversity
3. **Deduplicate** — exact + near-duplicate (MinHash)
4. **Train** — downstream model on synthetic vs real vs mix
5. **Evaluate** — metrics comparison + gap analysis

## Results (example: intent classification)

| Data | Accuracy | F1 macro |
|---|---|---|
| Real only (200) | 0.87 | 0.85 |
| Synthetic only (2000) | 0.81 | 0.78 |
| Real + Synthetic | **0.92** | **0.90** |

Synthetic data boosts performance by +5% when mixed with real.

## Stack

- LLM generation: OpenAI API / local Mistral
- Validation: Pydantic + custom rules + Great Expectations
- Dedup: datasketch (MinHash LSH)
- Training: scikit-learn / transformers
- UI: Streamlit

## Quick start

\`\`\`bash
git clone https://github.com/username/synthetic-data-factory
cd synthetic-data-factory
pip install -r requirements.txt

# 1. Generate
python scripts/generate.py --config configs/config.yaml

# 2. Validate
python scripts/validate.py --input data/synthetic.jsonl

# 3. Train
python scripts/train.py --data data/synthetic.jsonl --output outputs/model

# 4. UI
streamlit run app.py
\`\`\`

## Project structure

See [docs/architecture.md](docs/architecture.md).

## What I learned

- LLM-generated data needs strict schema validation — 15-20% of raw outputs are invalid.
- Near-duplicate removal is critical: MinHash removes ~10% additional samples.
- Mixing real + synthetic beats either alone — 30% real + 70% synthetic is optimal.

## Next steps

- Add RLHF-style filtering (reward model scores samples)
- Curriculum: easy synthetic → hard synthetic
- Active learning loop with human-in-the-loop
