#!/usr/bin/env python3
"""Validate factor researcher JSON artifacts without calling a model or network."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = SCRIPT_DIR / "run_researcher.py"


def load_runner():
    module_spec = importlib.util.spec_from_file_location("factor_idea_researcher_runner", RUNNER_PATH)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("unable to load run_researcher.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate first-principles and factor-specification artifacts locally."
    )
    parser.add_argument("--first-principles", required=True)
    parser.add_argument("--factor-specification", required=True)
    parser.add_argument("--feature-dictionary")
    return parser.parse_args(argv)


def validate(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    runner = load_runner()
    first = load_object(Path(args.first_principles), "first-principles artifact")
    factor = load_object(Path(args.factor_specification), "factor-specification artifact")
    dictionary_path = (
        Path(args.feature_dictionary) if args.feature_dictionary
        else runner.DEFAULT_FEATURE_DICTIONARY
    )
    dictionary = runner.load_feature_dictionary(dictionary_path)
    feature_names = set(dictionary["features"])
    timeframe = dictionary["timeframe"] if args.feature_dictionary else None
    runner.validate_first_principles(first, feature_names)
    runner.validate_factor_spec(factor, feature_names, timeframe)
    return {
        "success": True,
        "feature_mode": "constrained",
        "feature_dictionary_origin": "provided" if args.feature_dictionary else "skill_default",
        "factor_name": factor["factor_name"],
        "missing_features": factor["missing_features"],
        "checks": {
            "first_principles_contract": "passed",
            "factor_specification_contract": "passed",
            "feature_dictionary_constraint": "passed",
        },
    }


def main(argv: list[str] | None = None) -> int:
    try:
        result = validate(argv)
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
