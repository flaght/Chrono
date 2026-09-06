#!/usr/bin/env python3
"""Deterministic time-series and cross-sectional factor evaluation."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import operator
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable


SCHEMA_VERSION = "1.0.0"
COMPARATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


class EvaluationError(RuntimeError):
    """Expected input or evaluation error."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute or load factor values, then run an explicit time-series or cross-sectional evaluation."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Prepared CSV, JSON, JSONL, Parquet, Feather, or IPC")
    source.add_argument("--factor-artifact", help="Validated factor artifact used to compute values")
    parser.add_argument("--market-data", help="Raw CSV, Parquet, Feather, or IPC for code mode")
    parser.add_argument("--return-data", help="Aligned future-return data for code mode")
    parser.add_argument("--factor-column", help="Required for prepared mode; inferred in code mode")
    parser.add_argument("--return-column", required=True)
    parser.add_argument(
        "--evaluation-type", required=True, choices=["time_series", "cross_section"]
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-observations", type=int, default=100)
    parser.add_argument("--roll-win", type=int, default=252)
    parser.add_argument("--min-periods", type=int, default=5)
    parser.add_argument("--min-cross-section-size", type=int, default=2)
    parser.add_argument(
        "--time-series-evaluator",
        help=(
            "Path to cux001.py. Required for time_series unless "
            "FACTOR_TIME_SERIES_EVALUATOR is set; FactorEvaluate1.run(), "
            "plot_results(), and save_results() are called."
        ),
    )
    parser.add_argument("--resampling-win", type=int, default=1)
    parser.add_argument("--fee", type=float, default=0.0003)
    parser.add_argument(
        "--scale-method",
        choices=["roll_min_max", "roll_zscore", "roll_quantile", "ew_zscore", "train_const", "raw"],
        default="roll_min_max",
    )
    parser.add_argument("--annualization-factor", type=int, default=252)
    parser.add_argument("--pass-rule", help="JSON file with metric, operator, and value")
    args = parser.parse_args(argv)
    if args.input:
        if not args.factor_column:
            parser.error("--factor-column is required with --input")
        if args.market_data or args.return_data:
            parser.error("--market-data and --return-data are only valid with --factor-artifact")
    else:
        if not args.market_data or not args.return_data:
            parser.error("--market-data and --return-data are required with --factor-artifact")
    if args.min_observations < 2:
        parser.error("--min-observations must be >= 2")
    if args.roll_win < 2:
        parser.error("--roll-win must be >= 2")
    if not 2 <= args.min_periods <= args.roll_win:
        parser.error("--min-periods must be between 2 and --roll-win")
    if args.min_cross_section_size < 2:
        parser.error("--min-cross-section-size must be >= 2")
    if args.resampling_win < 1:
        parser.error("--resampling-win must be >= 1")
    if args.annualization_factor < 1:
        parser.error("--annualization-factor must be >= 1")
    if args.evaluation_type == "time_series":
        configured = args.time_series_evaluator or os.environ.get(
            "FACTOR_TIME_SERIES_EVALUATOR"
        )
        if not configured:
            parser.error(
                "time_series requires --time-series-evaluator /path/to/cux001.py "
                "or FACTOR_TIME_SERIES_EVALUATOR"
            )
        args.time_series_evaluator = configured
    return args


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise EvaluationError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be a JSON object")
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvaluationError(f"input data not found: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                return [dict(row) for row in csv.DictReader(stream)]
        if suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
                raise EvaluationError("JSON input must be an array of objects")
            return value
        if suffix in {".jsonl", ".ndjson"}:
            rows = []
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise EvaluationError(f"JSONL line {number} must be an object")
                rows.append(value)
            return rows
        if suffix in {".parquet", ".feather", ".ipc"}:
            try:
                import polars as pl
            except ImportError as exc:
                raise EvaluationError(
                    f"reading {suffix} requires polars; install it or provide CSV/JSON"
                ) from exc
            frame = pl.read_parquet(path) if suffix == ".parquet" else pl.read_ipc(path)
            return frame.to_dicts()
    except EvaluationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
        raise EvaluationError(f"failed to read input data: {exc}") from exc
    raise EvaluationError(f"unsupported input format: {suffix or '<none>'}")


def finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def prepare_rows(
    rows: Iterable[dict[str, Any]], factor_column: str, return_column: str
) -> tuple[list[dict[str, Any]], int]:
    required = {"trade_time", "code", factor_column, return_column}
    prepared: list[dict[str, Any]] = []
    raw_count = 0
    for raw_count, row in enumerate(rows, 1):
        missing = sorted(required - set(row))
        if missing:
            raise EvaluationError(f"input row {raw_count} missing columns: {missing}")
        factor = finite_float(row[factor_column])
        future_return = finite_float(row[return_column])
        trade_time = str(row["trade_time"]).strip()
        code = str(row["code"]).strip()
        if factor is None or future_return is None or not trade_time or not code:
            continue
        prepared.append({
            "trade_time": trade_time,
            "code": code,
            "factor": factor,
            "future_return": future_return,
        })
    if raw_count == 0:
        raise EvaluationError("input data is empty")
    if not prepared:
        raise EvaluationError("input has no valid aligned factor/return observations")
    return prepared, raw_count


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    if denominator == 0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for index in range(start, end):
            ranks[ordered[index][0]] = rank
        start = end
    return ranks


def summarize_ic(
    ic_values: list[float], observations: int, *, total_ic: float | None
) -> dict[str, float | int | None]:
    ic_mean = fmean(ic_values) if ic_values else 0.0
    ic_std = stdev(ic_values) if len(ic_values) >= 2 else 0.0
    ic_sharpe = ic_mean / ic_std if ic_std > 0 else 0.0
    return {
        "total_ic": total_ic,
        "ic_mean": ic_mean,
        "abs_ic_mean": abs(ic_mean),
        "ic_std": ic_std,
        "ic_sharpe": ic_sharpe,
        "observations": observations,
        "period_count": len(ic_values),
    }


def evaluate_time_series(
    rows: list[dict[str, Any]], *, roll_win: int, min_periods: int
) -> dict[str, float | int | None]:
    codes = sorted({row["code"] for row in rows})
    if len(codes) != 1:
        raise EvaluationError(
            f"time_series evaluation requires exactly one code; found {len(codes)}"
        )
    ordered = sorted(rows, key=lambda row: row["trade_time"])
    factors = [row["factor"] for row in ordered]
    returns = [row["future_return"] for row in ordered]
    rolling: list[float] = []
    for end in range(min_periods, len(ordered) + 1):
        start = max(0, end - roll_win)
        if end - start < min_periods:
            continue
        value = pearson(factors[start:end], returns[start:end])
        if value is not None:
            rolling.append(value)
    return summarize_ic(rolling, len(ordered), total_ic=pearson(factors, returns))


def load_time_series_evaluator(path: Path):
    if not path.is_file():
        raise EvaluationError(f"time-series evaluator not found: {path}")
    os.environ.setdefault("MPLBACKEND", "Agg")
    spec = importlib.util.spec_from_file_location("factor_time_series_cux001", path)
    if spec is None or spec.loader is None:
        raise EvaluationError(f"unable to load time-series evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvaluationError(f"failed to import time-series evaluator: {exc}") from exc
    evaluator = getattr(module, "FactorEvaluatePolars", None)
    if evaluator is not None:
        return evaluator, "polars"
    evaluator = getattr(module, "FactorEvaluate1", None)
    if evaluator is not None:
        return evaluator, "pandas"
    raise EvaluationError(
        "time-series evaluator must export FactorEvaluatePolars or FactorEvaluate1"
    )


def evaluator_period_count(evaluator) -> int:
    polars_data = getattr(evaluator, "resample_data_pl", None)
    if polars_data is not None:
        try:
            return int(polars_data.get_column("ic").drop_nulls().len())
        except Exception as exc:
            raise EvaluationError(f"unable to count Polars rolling IC periods: {exc}") from exc
    pandas_data = getattr(evaluator, "resample_data", None)
    if pandas_data is not None:
        try:
            return int(pandas_data["ic"].notna().sum())
        except Exception as exc:
            raise EvaluationError(f"unable to count pandas rolling IC periods: {exc}") from exc
    raise EvaluationError(
        "time-series evaluator produced neither resample_data_pl nor resample_data"
    )


def run_complete_time_series_evaluation(
    rows: list[dict[str, Any]], *, factor_name: str, return_name: str,
    evaluator_path: Path, roll_win: int, resampling_win: int, fee: float,
    scale_method: str, annualization_factor: int,
):
    codes = sorted({row["code"] for row in rows})
    if len(codes) != 1:
        raise EvaluationError(
            f"time_series evaluation requires exactly one code; found {len(codes)}"
        )
    values = {
        "trade_time": [row["trade_time"] for row in rows],
        factor_name: [row["factor"] for row in rows],
        return_name: [row["future_return"] for row in rows],
    }
    evaluator_class, backend = load_time_series_evaluator(evaluator_path)
    if backend == "polars":
        try:
            import polars as pl
        except ImportError as exc:
            raise EvaluationError("FactorEvaluatePolars requires polars") from exc
        frame = pl.DataFrame(values)
    else:
        try:
            import pandas as pd
        except ImportError as exc:
            raise EvaluationError("FactorEvaluate1 requires pandas") from exc
        frame = pd.DataFrame(values)
    try:
        evaluator = evaluator_class(
            factor_data=frame,
            resampling_win=resampling_win,
            factor_name=factor_name,
            ret_name=return_name,
            roll_win=roll_win,
            fee=fee,
            scale_method=scale_method,
            annualization_factor=annualization_factor,
            name=factor_name,
        )
        stats = evaluator.run(is_check=True)
    except Exception as exc:
        raise EvaluationError(f"cux001 time-series evaluation failed: {exc}") from exc
    if not isinstance(stats, dict):
        raise EvaluationError("cux001 FactorEvaluate1.run() must return a dict")
    normalized = {
        key: finite_float(value) if not isinstance(value, (str, bool)) else value
        for key, value in stats.items()
    }
    normalized.update({
        "abs_ic_mean": abs(normalized.get("ic_mean") or 0.0),
        "ic_sharpe": normalized.get("ic_ir") or 0.0,
        "observations": len(rows),
        "period_count": evaluator_period_count(evaluator),
        "evaluator_backend": backend,
    })
    return normalized, evaluator


def evaluate_cross_section(
    rows: list[dict[str, Any]], *, min_cross_section_size: int
) -> dict[str, float | int | None]:
    codes = {row["code"] for row in rows}
    if len(codes) < 2:
        raise EvaluationError(
            f"cross_section evaluation requires at least two codes; found {len(codes)}"
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["trade_time"]].append(row)
    ic_values: list[float] = []
    observations = 0
    for trade_time in sorted(grouped):
        section = grouped[trade_time]
        if len(section) < min_cross_section_size:
            continue
        factors = [row["factor"] for row in section]
        returns = [row["future_return"] for row in section]
        value = pearson(average_ranks(factors), average_ranks(returns))
        if value is not None:
            ic_values.append(value)
            observations += len(section)
    return summarize_ic(ic_values, observations, total_ic=None)


def validate_factor_artifact(path: Path) -> dict[str, Any]:
    artifact = load_json_object(path, "factor artifact")
    if artifact.get("status") != "validated" or artifact.get("ready_for_evaluation") is not True:
        raise EvaluationError(
            "factor artifact must have status='validated' and ready_for_evaluation=true"
        )
    return artifact


def resolve_factor_artifact(path: Path) -> tuple[dict[str, Any], Path]:
    artifact = validate_factor_artifact(path)
    factor_name = artifact.get("factor_name")
    relative = artifact.get("factor_file")
    expected_hash = artifact.get("factor_sha256")
    required_fields = artifact.get("required_input_fields")
    if not isinstance(factor_name, str) or not factor_name:
        raise EvaluationError("factor artifact factor_name is invalid")
    if not isinstance(relative, str) or not relative:
        raise EvaluationError("factor artifact factor_file is invalid")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise EvaluationError("factor artifact factor_sha256 is invalid")
    if not isinstance(required_fields, list) or any(
        not isinstance(item, str) or not item for item in required_fields
    ):
        raise EvaluationError("factor artifact required_input_fields is invalid")
    if not {"trade_time", "code"}.issubset(required_fields):
        raise EvaluationError("factor artifact required_input_fields must include trade_time and code")
    root = path.resolve().parent
    factor_path = (root / relative).resolve()
    try:
        factor_path.relative_to(root)
    except ValueError as exc:
        raise EvaluationError("factor artifact factor_file escapes its run directory") from exc
    if not factor_path.is_file():
        raise EvaluationError(f"validated factor code not found: {factor_path}")
    if sha256_file(factor_path) != expected_hash:
        raise EvaluationError("validated factor code SHA256 does not match artifact")
    return artifact, factor_path


def require_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise EvaluationError("factor computation mode requires polars") from exc
    return pl


def scan_lazy(path: Path, pl, label: str):
    if not path.is_file():
        raise EvaluationError(f"{label} not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.scan_csv(path)
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix in {".feather", ".ipc"}:
        return pl.scan_ipc(path)
    raise EvaluationError(f"unsupported {label} format: {suffix or '<none>'}")


def lazy_columns(frame) -> list[str]:
    try:
        return list(frame.collect_schema().names())
    except AttributeError:
        return list(frame.schema)


def load_factor_compute(path: Path):
    spec = importlib.util.spec_from_file_location(f"evaluated_factor_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise EvaluationError(f"unable to load validated factor code: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EvaluationError(f"failed to import validated factor code: {exc}") from exc
    compute = getattr(module, "compute", None)
    if not callable(compute):
        raise EvaluationError("validated factor code has no callable compute")
    return compute


def compute_and_align(
    artifact_path: Path, market_path: Path, return_path: Path, return_column: str
) -> tuple[list[dict[str, Any]], int, dict[str, Any], Any, Any, str]:
    artifact, factor_path = resolve_factor_artifact(artifact_path)
    pl = require_polars()
    factor_name = artifact["factor_name"]
    required_fields = artifact["required_input_fields"]
    market = scan_lazy(market_path, pl, "market data")
    missing_market = sorted(set(required_fields) - set(lazy_columns(market)))
    if missing_market:
        raise EvaluationError(f"market data missing required fields: {missing_market}")
    returns = scan_lazy(return_path, pl, "return data")
    missing_returns = sorted({"trade_time", "code", return_column} - set(lazy_columns(returns)))
    if missing_returns:
        raise EvaluationError(f"return data missing required fields: {missing_returns}")
    try:
        factor_lazy = load_factor_compute(factor_path)(market.select(required_fields))
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(f"factor compute failed before collection: {exc}") from exc
    if not isinstance(factor_lazy, pl.LazyFrame):
        raise EvaluationError("factor compute must return polars.LazyFrame")
    try:
        factor_frame = factor_lazy.collect()
    except Exception as exc:
        raise EvaluationError(f"factor value collection failed: {exc}") from exc
    expected = ["trade_time", "code", factor_name]
    if factor_frame.columns != expected:
        raise EvaluationError(f"factor output columns must be {expected!r}; found {factor_frame.columns!r}")
    duplicate_factor_keys = (
        factor_frame.lazy().group_by(["trade_time", "code"]).len()
        .filter(pl.col("len") > 1).select(pl.len()).collect().item()
    )
    if duplicate_factor_keys:
        raise EvaluationError(f"factor output contains {duplicate_factor_keys} duplicate key group(s)")
    try:
        return_frame = returns.select(["trade_time", "code", return_column]).collect()
    except Exception as exc:
        raise EvaluationError(f"return data collection failed: {exc}") from exc
    duplicate_return_keys = (
        return_frame.lazy().group_by(["trade_time", "code"]).len()
        .filter(pl.col("len") > 1).select(pl.len()).collect().item()
    )
    if duplicate_return_keys:
        raise EvaluationError(f"return data contains {duplicate_return_keys} duplicate key group(s)")
    aligned = factor_frame.join(return_frame, on=["trade_time", "code"], how="inner")
    if aligned.height == 0:
        raise EvaluationError("factor values and return data have no matching trade_time/code keys")
    report = {
        "status": "completed",
        "factor_name": factor_name,
        "factor_rows": factor_frame.height,
        "return_rows": return_frame.height,
        "aligned_rows": aligned.height,
        "market_data_sha256": sha256_file(market_path),
        "return_data_sha256": sha256_file(return_path),
        "factor_code_sha256": sha256_file(factor_path),
        "data_isolation": "factor compute received only required_input_fields; future returns were joined afterward",
    }
    return aligned.to_dicts(), aligned.height, report, factor_frame, aligned, factor_name


def apply_pass_rule(metrics: dict[str, Any], path: Path) -> dict[str, Any]:
    rule = load_json_object(path, "pass rule")
    metric = rule.get("metric")
    comparison = rule.get("operator")
    threshold = finite_float(rule.get("value"))
    if not isinstance(metric, str) or metric not in metrics:
        raise EvaluationError(f"pass rule metric is unavailable: {metric!r}")
    if comparison not in COMPARATORS:
        raise EvaluationError(f"unsupported pass rule operator: {comparison!r}")
    actual = finite_float(metrics[metric])
    if threshold is None or actual is None:
        raise EvaluationError("pass rule metric and value must be finite numbers")
    passed = COMPARATORS[comparison](actual, threshold)
    return {
        "metric": metric,
        "operator": comparison,
        "threshold": threshold,
        "actual": actual,
        "passed": passed,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise EvaluationError(f"output directory already exists: {output_dir}")
    computation_report = None
    factor_frame = None
    aligned_frame = None
    if args.input:
        input_path = Path(args.input)
        factor_column = args.factor_column
        rows, raw_count = prepare_rows(
            load_rows(input_path), factor_column, args.return_column
        )
        input_hashes = {"prepared_input_sha256": sha256_file(input_path)}
        input_mode = "prepared_values"
    else:
        artifact_path = Path(args.factor_artifact)
        market_path = Path(args.market_data)
        return_path = Path(args.return_data)
        computed, raw_count, computation_report, factor_frame, aligned_frame, inferred = (
            compute_and_align(artifact_path, market_path, return_path, args.return_column)
        )
        if args.factor_column and args.factor_column != inferred:
            raise EvaluationError(
                f"--factor-column {args.factor_column!r} does not match artifact factor_name {inferred!r}"
            )
        factor_column = inferred
        rows, _ = prepare_rows(computed, factor_column, args.return_column)
        input_hashes = {
            "factor_artifact_sha256": sha256_file(artifact_path),
            "market_data_sha256": computation_report["market_data_sha256"],
            "return_data_sha256": computation_report["return_data_sha256"],
            "factor_code_sha256": computation_report["factor_code_sha256"],
        }
        input_mode = "validated_code"
    time_series_evaluator = None
    if args.evaluation_type == "time_series":
        metrics, time_series_evaluator = run_complete_time_series_evaluation(
            rows,
            factor_name=factor_column,
            return_name=args.return_column,
            evaluator_path=Path(args.time_series_evaluator),
            roll_win=args.roll_win,
            resampling_win=args.resampling_win,
            fee=args.fee,
            scale_method=args.scale_method,
            annualization_factor=args.annualization_factor,
        )
    else:
        metrics = evaluate_cross_section(
            rows, min_cross_section_size=args.min_cross_section_size
        )
    if metrics["observations"] < args.min_observations:
        raise EvaluationError(
            f"valid observations {metrics['observations']} below minimum {args.min_observations}"
        )
    if metrics["period_count"] == 0:
        raise EvaluationError("evaluation produced no valid IC periods")

    pass_result = apply_pass_rule(metrics, Path(args.pass_rule)) if args.pass_rule else None
    status = "completed"
    if pass_result is not None:
        status = "passed" if pass_result["passed"] else "failed_rule"
    request_record = {
        "input_mode": input_mode,
        "input": str(Path(args.input).resolve()) if args.input else None,
        "market_data": str(Path(args.market_data).resolve()) if args.market_data else None,
        "return_data": str(Path(args.return_data).resolve()) if args.return_data else None,
        "factor_column": factor_column,
        "return_column": args.return_column,
        "evaluation_type": args.evaluation_type,
        "min_observations": args.min_observations,
        "roll_win": args.roll_win if args.evaluation_type == "time_series" else None,
        "min_periods": args.min_periods if args.evaluation_type == "time_series" else None,
        "time_series_evaluator": (
            str(Path(args.time_series_evaluator).resolve())
            if args.evaluation_type == "time_series" else None
        ),
        "resampling_win": args.resampling_win if args.evaluation_type == "time_series" else None,
        "fee": args.fee if args.evaluation_type == "time_series" else None,
        "scale_method": args.scale_method if args.evaluation_type == "time_series" else None,
        "annualization_factor": (
            args.annualization_factor if args.evaluation_type == "time_series" else None
        ),
        "min_cross_section_size": (
            args.min_cross_section_size if args.evaluation_type == "cross_section" else None
        ),
        "factor_artifact": (
            str(Path(args.factor_artifact).resolve()) if args.factor_artifact else None
        ),
        "pass_rule": str(Path(args.pass_rule).resolve()) if args.pass_rule else None,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "evaluation_type": args.evaluation_type,
        "input_mode": input_mode,
        "factor_column": factor_column,
        "return_column": args.return_column,
        "metrics": metrics,
        "pass_rule": pass_result,
    }
    report = {
        "status": status,
        "input_rows": raw_count,
        "valid_rows": len(rows),
        "dropped_rows": raw_count - len(rows),
        "unique_codes": len({row["code"] for row in rows}),
        "unique_times": len({row["trade_time"] for row in rows}),
        "factor_artifact_checked": input_mode == "validated_code",
        "warnings": ([
            "prepared_values 模式的未来收益时间对齐由上游负责；本评估不能独立证明不存在未来数据泄漏。"
        ] if input_mode == "prepared_values" else []),
    }

    artifacts: dict[str, Any] = {
        "request.json": request_record,
        "evaluation-result.json": result,
        "evaluation-report.json": report,
    }
    if pass_result is not None:
        artifacts["pass-rule-result.json"] = pass_result
    if computation_report is not None:
        artifacts["computation-report.json"] = computation_report
        output_dir.mkdir(parents=True, exist_ok=False)
        factor_frame.write_parquet(output_dir / "factor-values.parquet")
        aligned_frame.write_parquet(output_dir / "aligned-evaluation-data.parquet")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    if time_series_evaluator is not None:
        try:
            time_series_evaluator.plot_results()
            time_series_evaluator.save_results(str(output_dir))
        except Exception as exc:
            raise EvaluationError(f"cux001 save_results failed: {exc}") from exc
        complete_dir = output_dir / factor_column
        report["complete_time_series_results"] = {
            "directory": str(complete_dir.resolve()),
            "artifacts": [
                "performance_summary.txt",
                "nav.csv",
                "ic.csv",
                "turnover.csv",
                "evaluation_plot.png",
                "evaluation.xml",
            ],
        }
    for name, value in artifacts.items():
        (output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "evaluation_type": args.evaluation_type,
        "input_mode": input_mode,
        "input_hashes": input_hashes,
        "complete_time_series_results": (
            str((output_dir / factor_column).resolve())
            if time_series_evaluator is not None else None
        ),
        "artifacts": [
            *artifacts.keys(),
            *(["factor-values.parquet", "aligned-evaluation-data.parquet"] if computation_report else []),
            "manifest.json",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "success": True,
        "status": status,
        "evaluation_type": args.evaluation_type,
        "metrics": metrics,
        "output_dir": str(output_dir.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
