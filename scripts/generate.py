"""CLI: генерация синтетического датасета."""
import argparse
import yaml
from src.generator import SyntheticGenerator
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(42)
    gen = SyntheticGenerator(
        model=cfg["generation"]["model"],
        temperature=cfg["generation"]["temperature"],
    )
    gen.generate_dataset(
        specs=cfg["generation"]["specs"],
        output_path=cfg["generation"]["output"],
    )


if __name__ == "__main__":
    main()
