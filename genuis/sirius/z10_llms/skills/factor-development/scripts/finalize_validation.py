#!/usr/bin/env python3
"""Finalize an approved factor candidate with a controlled LazyFrame runtime test."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATOR_PATH = SCRIPT_DIR / "validate_factor.py"


class FinalizationError(RuntimeError):
    """Expected approval, artifact, dependency, or runtime validation error."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a statically validated factor on test data and create validated-factor-artifact.json."
    )
    parser.add_argument("--run-dir", required=True, help="Output directory created by run_developer.py")
    parser.add_argument(
        "--approval",
        help="Optional human approval JSON; omit for runtime-only validation",
    )
    parser.add_argument("--test-data", required=True, help="CSV, Parquet, Feather, or IPC test data")
    return parser.parse_args(argv)


def load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FinalizationError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"{label} must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_approval(approval: dict[str, Any], factor_sha256: str) -> None:
    if approval.get("approved") is not True:
        raise FinalizationError("approval.approved must be true")
    reviewer = approval.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise FinalizationError("approval.reviewer must be a non-empty string")
    if approval.get("factor_sha256") != factor_sha256:
        raise FinalizationError("approval.factor_sha256 does not match the candidate code")


def load_validator():
    module_spec = importlib.util.spec_from_file_location("factor_static_validator", VALIDATOR_PATH)
    if module_spec is None or module_spec.loader is None:
        raise FinalizationError("unable to load static factor validator")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.validate


def scan_frame(path: Path, pl):
    if not path.is_file():
        raise FinalizationError(f"test data not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.scan_csv(path)
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix in {".feather", ".ipc"}:
        return pl.scan_ipc(path)
    raise FinalizationError(f"unsupported test data format: {suffix or '<none>'}")


def frame_columns(lazy_frame) -> list[str]:
    try:
        return list(lazy_frame.collect_schema().names())
    except AttributeError:
        return list(lazy_frame.schema)


def load_factor_module(path: Path):
    module_spec = importlib.util.spec_from_file_location(
        f"validated_factor_{path.stem}", path
    )
    if module_spec is None or module_spec.loader is None:
        raise FinalizationError(f"unable to load factor module: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    compute = getattr(module, "compute", None)
    if not callable(compute):
        raise FinalizationError("factor module does not expose callable compute")
    return module


def runtime_validate(
    run_dir: Path, approval_path: Path | None, test_data: Path
) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    report_path = run_dir / "runtime-validation-report.json"
    artifact_path = run_dir / "validated-factor-artifact.json"
    if report_path.exists() or artifact_path.exists():
        raise FinalizationError("runtime validation artifacts already exist; use a new development run")
    manifest = load_object(manifest_path, "development manifest")
    generation = load_object(run_dir / "generation-report.json", "generation report")
    factor_relative = manifest.get("factor_file")
    expected_hash = manifest.get("factor_sha256")
    if not isinstance(factor_relative, str) or not factor_relative:
        raise FinalizationError("development manifest factor_file is missing")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise FinalizationError("development manifest factor_sha256 is missing")
    factor_path = (run_dir / factor_relative).resolve()
    try:
        factor_path.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise FinalizationError("factor_file escapes the development run directory") from exc
    if not factor_path.is_file():
        raise FinalizationError(f"candidate factor not found: {factor_path}")
    actual_hash = sha256_file(factor_path)
    if actual_hash != expected_hash:
        raise FinalizationError("candidate factor SHA256 differs from development manifest")
    approval = None
    if approval_path is not None:
        approval = load_object(approval_path, "approval")
        validate_approval(approval, actual_hash)
    validation_mode = "human_approved" if approval is not None else "runtime_only"

    static_errors = load_validator()(factor_path)
    if static_errors:
        raise FinalizationError("static validation failed: " + "; ".join(static_errors))
    try:
        compile(factor_path.read_text(encoding="utf-8"), str(factor_path), "exec")
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise FinalizationError(f"factor compile validation failed: {exc}") from exc
    try:
        import polars as pl
    except ImportError as exc:
        raise FinalizationError("runtime validation requires polars") from exc

    required_fields = generation.get("required_input_fields")
    factor_name = generation.get("factor_name")
    if not isinstance(required_fields, list) or any(
        not isinstance(item, str) for item in required_fields
    ):
        raise FinalizationError("generation report required_input_fields is invalid")
    if not isinstance(factor_name, str) or not factor_name:
        raise FinalizationError("generation report factor_name is invalid")
    for key in ("trade_time", "code"):
        if key not in required_fields:
            required_fields.insert(0 if key == "trade_time" else 1, key)

    source = scan_frame(test_data, pl)
    missing = sorted(set(required_fields) - set(frame_columns(source)))
    if missing:
        raise FinalizationError(f"test data missing required fields: {missing}")
    module = load_factor_module(factor_path)
    try:
        result_lazy = module.compute(source.select(required_fields))
    except Exception as exc:
        raise FinalizationError(f"factor compute failed before collection: {exc}") from exc
    if not isinstance(result_lazy, pl.LazyFrame):
        raise FinalizationError("factor compute must return polars.LazyFrame")
    try:
        result = result_lazy.collect()
    except Exception as exc:
        raise FinalizationError(f"factor LazyFrame collection failed: {exc}") from exc
    expected_columns = ["trade_time", "code", factor_name]
    if result.columns != expected_columns:
        raise FinalizationError(
            f"factor output columns must be {expected_columns!r}; found {result.columns!r}"
        )
    if result.height == 0:
        raise FinalizationError("factor runtime output is empty")
    duplicate_count = (
        result.lazy().group_by(["trade_time", "code"]).len()
        .filter(pl.col("len") > 1).select(pl.len()).collect().item()
    )
    if duplicate_count:
        raise FinalizationError(f"factor output contains {duplicate_count} duplicate key group(s)")
    finite_count = result.select(
        pl.col(factor_name).cast(pl.Float64, strict=False).is_finite().sum()
    ).item()
    if not finite_count:
        raise FinalizationError("factor output has no finite factor values")

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    runtime_report = {
        "status": "passed",
        "validation_mode": validation_mode,
        "factor_name": factor_name,
        "factor_sha256": actual_hash,
        "test_data": str(test_data.resolve()),
        "test_data_sha256": sha256_file(test_data),
        "output_rows": result.height,
        "finite_factor_values": int(finite_count),
        "duplicate_key_groups": int(duplicate_count),
        "checks": {
            "candidate_hash": "passed",
            "human_approval": "approved" if approval is not None else "not_required",
            "static_contract": "passed",
            "python_compile": "passed",
            "runtime_lazyframe": "passed",
        },
    }
    artifact = {
        "schema_version": "1.0.0",
        "created_at": now,
        "status": "validated",
        "ready_for_evaluation": True,
        "validation_mode": validation_mode,
        "factor_name": factor_name,
        "factor_file": factor_relative,
        "factor_sha256": actual_hash,
        "batch": generation.get("batch"),
        "max_window": generation.get("max_window"),
        "required_input_fields": required_fields,
        "approval": ({
            "reviewer": approval["reviewer"],
            "approved_at": approval.get("approved_at"),
        } if approval is not None else None),
        "validation": runtime_report["checks"],
    }
    report_path.write_text(json.dumps(runtime_report, ensure_ascii=False, indent=2), encoding="utf-8")
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["status"] = "validated"
    manifest["ready_for_evaluation"] = True
    manifest["artifacts"] = list(dict.fromkeys([
        *manifest.get("artifacts", []),
        "runtime-validation-report.json",
        "validated-factor-artifact.json",
    ]))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = runtime_validate(
        Path(args.run_dir),
        Path(args.approval) if args.approval else None,
        Path(args.test_data),
    )
    print(json.dumps({
        "success": True,
        "status": artifact["status"],
        "ready_for_evaluation": artifact["ready_for_evaluation"],
        "validation_mode": artifact["validation_mode"],
        "artifact": str((Path(args.run_dir) / "validated-factor-artifact.json").resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FinalizationError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
