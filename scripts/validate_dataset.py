#!/usr/bin/env python3
"""CLI to validate Taboo datasets for secret leakage."""

from __future__ import annotations

import argparse
from pathlib import Path

from taboo_gwen3.config.secrets import load_secret_policy
from taboo_gwen3.data.validators import SecretLeakValidator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Taboo datasets for leakage")
    parser.add_argument("policy", type=Path, help="Path to secrets YAML definition")
    parser.add_argument(
        "datasets",
        nargs="+",
        type=Path,
        help="Paths to JSONL datasets with chat messages",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = load_secret_policy(args.policy)
    validator = SecretLeakValidator(policy=policy)
    validator.assert_clean(args.datasets)
    print("All datasets passed the leakage checks.")


if __name__ == "__main__":
    main()
