"""CLI: валидация датасета."""
import argparse
import json
from src.utils import load_jsonl
from src.validator import DataValidator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/validation_report.json")
    args = parser.parse_args()

    samples = load_jsonl(args.input)
    validator = DataValidator(samples)
    report = validator.report()

    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
