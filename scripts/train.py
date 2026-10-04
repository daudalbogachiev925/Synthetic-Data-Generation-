"""CLI: обучение downstream-модели."""
import argparse
from sklearn.model_selection import train_test_split
from src.utils import load_jsonl, save_jsonl
from src.trainer import train_and_eval
from src.deduplicator import deduplicate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--real", default=None, help="опционально: реальные данные")
    parser.add_argument("--output", default="outputs/metrics.json")
    args = parser.parse_args()

    synthetic = load_jsonl(args.data)
    print(f"Loaded {len(synthetic)} synthetic samples")
    synthetic = deduplicate(synthetic)
    print(f"After dedup: {len(synthetic)}")

    # Разбиваем на train/test
    train, test = train_test_split(synthetic, test_size=0.2, random_state=42, stratify=[s.intent for s in synthetic])

    results = train_and_eval(train, test)
    print(f"Accuracy: {results['accuracy']}, F1 macro: {results['f1_macro']}")
    print(results["report"])


if __name__ == "__main__":
    main()
