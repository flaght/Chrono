#!/usr/bin/env python3
"""Resumable file-contract orchestrator for the KD-HYS factor pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"
SKILL_DIR = Path(__file__).resolve().parent.parent
KD_HYS_DIR = SKILL_DIR.parent
RESEARCH_SCRIPT = KD_HYS_DIR / "factor-idea-researcher" / "scripts" / "run_researcher.py"
DEVELOP_SCRIPT = KD_HYS_DIR / "factor-development" / "scripts" / "run_developer.py"
FINALIZE_SCRIPT = KD_HYS_DIR / "factor-development" / "scripts" / "finalize_validation.py"
EVALUATE_SCRIPT = KD_HYS_DIR / "factor-evaluation" / "scripts" / "run_evaluation.py"
SECRET_FRAGMENTS = ("api_key", "apikey", "token", "secret", "password")
DEFAULT_CONFIG = SKILL_DIR / "config" / "pipeline.defaults.json"
MAX_IDEA_BYTES = 1_000_000
MAX_IDEA_CHARS = 20_000
RESERVED_RUN_ENTRIES = {
    "request.snapshot.json",
    "research",
    "development",
    "evaluation",
}
FACTOR_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class OrchestrationError(RuntimeError):
    """Expected request, state, child-process, or artifact error."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume the KD-HYS factor pipeline.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--idea", help="Factor idea; remaining settings come from --config")
    source.add_argument("--input", help="UTF-8 .txt or .md file containing the factor idea")
    source.add_argument("--request", help="Pipeline request JSON; overrides --config")
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help="Reusable pipeline defaults JSON",
    )
    parser.add_argument("--factor-name", help="Override final generated factor name")
    parser.add_argument("--run-dir", required=True, help="New or resumable pipeline directory")
    return parser.parse_args(argv)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def request_digest(request: dict[str, Any]) -> str:
    payload = json.dumps(
        request, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_idea_file(path: Path) -> str:
    if not path.is_file():
        raise OrchestrationError(f"idea input not found: {path}")
    if path.suffix.lower() not in {".txt", ".md", ".markdown"}:
        raise OrchestrationError("idea input must be a .txt, .md, or .markdown file")
    if path.stat().st_size > MAX_IDEA_BYTES:
        raise OrchestrationError(f"idea input exceeds {MAX_IDEA_BYTES} bytes")
    try:
        idea = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise OrchestrationError(f"failed to read UTF-8 idea input: {exc}") from exc
    if not idea:
        raise OrchestrationError("idea input is empty")
    if len(idea) > MAX_IDEA_CHARS:
        raise OrchestrationError(f"factor idea exceeds {MAX_IDEA_CHARS} characters")
    return idea


def ensure_initializable_run_dir(run_dir: Path, manifest_path: Path) -> None:
    if not run_dir.exists() or manifest_path.is_file():
        return
    conflicts = sorted(
        entry.name for entry in run_dir.iterdir()
        if entry.name in RESERVED_RUN_ENTRIES
    )
    if conflicts:
        raise OrchestrationError(
            "run directory has no pipeline-manifest.json but contains reserved "
            f"pipeline entries: {conflicts}"
        )


def load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise OrchestrationError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OrchestrationError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise OrchestrationError(f"{label} must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_secrets(value: Any, prefix: str = "request") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in SECRET_FRAGMENTS):
                raise OrchestrationError(f"secret-like field is forbidden in request: {prefix}.{key}")
            reject_secrets(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_secrets(child, f"{prefix}[{index}]")


def section(request: dict[str, Any], name: str) -> dict[str, Any]:
    value = request.get(name)
    if not isinstance(value, dict):
        raise OrchestrationError(f"request.{name} must be an object")
    return value


def resolve_path(value: Any, base: Path, label: str, *, required: bool = False) -> Path | None:
    if value in (None, ""):
        if required:
            raise OrchestrationError(f"{label} is required")
        return None
    if not isinstance(value, str):
        raise OrchestrationError(f"{label} must be a path string")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def validate_execution(config: dict[str, Any], label: str) -> str:
    execution = config.get("execution", "codex")
    if execution not in {"codex", "python"}:
        raise OrchestrationError(f"{label}.execution must be codex or python")
    return execution


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def update_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
    status: str,
    *,
    stage: str,
    message: str,
) -> None:
    manifest["status"] = status
    manifest["current_stage"] = stage
    manifest["updated_at"] = now()
    manifest["next_action"] = message
    history = manifest.setdefault("history", [])
    if not history or history[-1].get("status") != status or history[-1].get("message") != message:
        history.append({"at": manifest["updated_at"], "stage": stage, "status": status, "message": message})
    write_json(manifest_path, manifest)


def run_child(command: list[str], stage: str) -> None:
    print(f"[{stage}] 启动：{' '.join(command[:3])} ...", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise OrchestrationError(f"{stage} child process failed with exit code {completed.returncode}")


def add_option(command: list[str], name: str, value: Any) -> None:
    if value not in (None, ""):
        command.extend([name, str(value)])


def validate_specification(path: Path) -> None:
    value = load_object(path, "factor specification")
    for field in ("factor_name", "formula_description", "required_features"):
        if field not in value:
            raise OrchestrationError(f"factor specification missing {field}")


def validate_artifact(path: Path) -> None:
    value = load_object(path, "validated factor artifact")
    if value.get("status") != "validated" or value.get("ready_for_evaluation") is not True:
        raise OrchestrationError("factor artifact is not ready for evaluation")


def run_pipeline(
    request_path: Path, run_dir: Path, *, request_override: dict[str, Any] | None = None,
    request_base: Path | None = None,
) -> dict[str, Any]:
    request_path = request_path.resolve()
    request_base = request_base.resolve() if request_base else request_path.parent
    request = request_override if request_override is not None else load_object(request_path, "pipeline request")
    reject_secrets(request)
    if request.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise OrchestrationError(f"unsupported request schema_version: {request.get('schema_version')!r}")
    research = section(request, "research")
    development = section(request, "development")
    evaluation = section(request, "evaluation")
    research_execution = validate_execution(research, "research")
    development_execution = validate_execution(development, "development")

    manifest_path = run_dir / "pipeline-manifest.json"
    ensure_initializable_run_dir(run_dir, manifest_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    request_hash = request_digest(request)
    if manifest_path.is_file():
        manifest = load_object(manifest_path, "pipeline manifest")
        if manifest.get("status") == "completed" and manifest.get("request_sha256") != request_hash:
            raise OrchestrationError("completed run cannot be resumed with a different request")
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": now(),
            "status": "initialized",
            "current_stage": "initialization",
            "history": [],
            "stages": {},
        }
    manifest["request"] = str(request_path)
    manifest["request_sha256"] = request_hash
    manifest["executions"] = {
        "research": research_execution,
        "development": development_execution,
        "evaluation": "python_deterministic",
    }
    write_json(run_dir / "request.snapshot.json", request)
    update_manifest(manifest_path, manifest, manifest.get("status", "initialized"), stage="initialization", message="请求已校验")

    research_dir = run_dir / "research"
    specification = resolve_path(research.get("specification"), request_base, "research.specification")
    if specification is None and (research_dir / "factor-specification.json").is_file():
        specification = research_dir / "factor-specification.json"
    if specification is None:
        if research_execution == "codex":
            update_manifest(manifest_path, manifest, "awaiting_codex_research", stage="research", message="请按 factor-idea-researcher 生成 factor-specification.json，并写入 research.specification")
            return manifest
        idea = research.get("idea")
        if not isinstance(idea, str) or not idea.strip():
            raise OrchestrationError("research.idea is required for Python research mode")
        mode = research.get("mode", "direct")
        if mode not in {"direct", "search", "fetch"}:
            raise OrchestrationError("research.mode must be direct, search, or fetch")
        update_manifest(manifest_path, manifest, "researching", stage="research", message="正在生成因子说明书")
        command = [sys.executable, str(RESEARCH_SCRIPT), "--idea", idea, "--mode", mode, "--output-dir", str(research_dir)]
        for url in research.get("urls", []):
            command.extend(["--url", str(url)])
        add_option(command, "--query", research.get("query"))
        add_option(command, "--feature-dictionary", resolve_path(research.get("feature_dictionary"), request_base, "research.feature_dictionary"))
        add_option(command, "--env-file", resolve_path(research.get("env_file"), request_base, "research.env_file"))
        add_option(command, "--provider", research.get("provider"))
        add_option(command, "--model", research.get("model"))
        add_option(command, "--timeout", research.get("timeout", 900))
        add_option(command, "--max-retries", research.get("max_retries", 2))
        run_child(command, "research")
        specification = research_dir / "factor-specification.json"
    validate_specification(specification)
    manifest["stages"]["research"] = {"status": "completed", "execution": research_execution, "specification": str(specification.resolve()), "sha256": sha256_file(specification)}

    development_dir = run_dir / "development"
    artifact = resolve_path(development.get("validated_factor_artifact"), request_base, "development.validated_factor_artifact")
    if artifact is None and (development_dir / "validated-factor-artifact.json").is_file():
        artifact = development_dir / "validated-factor-artifact.json"
    if artifact is None and not (development_dir / "manifest.json").is_file():
        if development_execution == "codex":
            update_manifest(manifest_path, manifest, "awaiting_codex_development", stage="development", message="请按 factor-development 开发并验证代码，再写入 development.validated_factor_artifact")
            return manifest
        update_manifest(manifest_path, manifest, "developing", stage="development", message="正在生成候选因子代码")
        command = [sys.executable, str(DEVELOP_SCRIPT), "--spec", str(specification), "--output-dir", str(development_dir)]
        development_dictionary = resolve_path(
            development.get("feature_dictionary"), request_base,
            "development.feature_dictionary",
        )
        if development_dictionary is None:
            research_dictionary = research_dir / "feature-dictionary.json"
            if research_dictionary.is_file():
                development_dictionary = research_dictionary
        add_option(command, "--feature-dictionary", development_dictionary)
        add_option(command, "--factor-name", development.get("factor_name"))
        add_option(command, "--env-file", resolve_path(development.get("env_file"), request_base, "development.env_file"))
        add_option(command, "--provider", development.get("provider"))
        add_option(command, "--model", development.get("model"))
        add_option(command, "--timeout", development.get("timeout", 900))
        add_option(command, "--max-retries", development.get("max_retries", 2))
        run_child(command, "development")
    if artifact is None:
        approval = resolve_path(development.get("approval"), request_base, "development.approval")
        test_data = resolve_path(development.get("test_data"), request_base, "development.test_data")
        if test_data is None:
            manifest["stages"]["development"] = {"status": "awaiting_runtime_validation", "execution": development_execution, "run_dir": str(development_dir.resolve())}
            update_manifest(manifest_path, manifest, "awaiting_runtime_validation", stage="development", message="请在请求中提供 development.test_data；development.approval 可选")
            return manifest
        update_manifest(manifest_path, manifest, "validating", stage="development", message="正在执行受控运行验证；人工审批为可选增强")
        command = [sys.executable, str(FINALIZE_SCRIPT), "--run-dir", str(development_dir), "--test-data", str(test_data)]
        if approval is not None:
            command.extend(["--approval", str(approval)])
        run_child(command, "development-validation")
        artifact = development_dir / "validated-factor-artifact.json"
    validate_artifact(artifact)
    manifest["stages"]["development"] = {"status": "completed", "execution": development_execution, "artifact": str(artifact.resolve()), "sha256": sha256_file(artifact)}

    evaluation_dir = run_dir / "evaluation"
    result_path = evaluation_dir / "evaluation-result.json"
    if not result_path.is_file():
        evaluation_type = evaluation.get("evaluation_type")
        if evaluation_type not in {"time_series", "cross_section"}:
            raise OrchestrationError("evaluation.evaluation_type must be time_series or cross_section")
        return_column = evaluation.get("return_column")
        if not isinstance(return_column, str) or not return_column:
            raise OrchestrationError("evaluation.return_column is required")
        update_manifest(manifest_path, manifest, "evaluating", stage="evaluation", message="正在计算因子值并执行评估")
        command = [sys.executable, str(EVALUATE_SCRIPT)]
        prepared = resolve_path(evaluation.get("input"), request_base, "evaluation.input")
        if prepared:
            factor_column = evaluation.get("factor_column")
            if not isinstance(factor_column, str) or not factor_column:
                raise OrchestrationError("evaluation.factor_column is required with evaluation.input")
            command.extend(["--input", str(prepared), "--factor-column", factor_column])
        else:
            market = resolve_path(evaluation.get("market_data"), request_base, "evaluation.market_data", required=True)
            returns = resolve_path(evaluation.get("return_data"), request_base, "evaluation.return_data", required=True)
            command.extend(["--factor-artifact", str(artifact), "--market-data", str(market), "--return-data", str(returns)])
        command.extend(["--return-column", return_column, "--evaluation-type", evaluation_type, "--output-dir", str(evaluation_dir)])
        if evaluation_type == "time_series":
            evaluator = resolve_path(
                evaluation.get("time_series_evaluator"), request_base,
                "evaluation.time_series_evaluator", required=True,
            )
            command.extend(["--time-series-evaluator", str(evaluator)])
        for key, option in (("min_observations", "--min-observations"), ("roll_win", "--roll-win"), ("min_periods", "--min-periods"), ("min_cross_section_size", "--min-cross-section-size"), ("resampling_win", "--resampling-win"), ("fee", "--fee"), ("scale_method", "--scale-method"), ("annualization_factor", "--annualization-factor")):
            add_option(command, option, evaluation.get(key))
        add_option(command, "--pass-rule", resolve_path(evaluation.get("pass_rule"), request_base, "evaluation.pass_rule"))
        run_child(command, "evaluation")
    result = load_object(result_path, "evaluation result")
    manifest["stages"]["evaluation"] = {"status": result.get("status"), "result": str(result_path.resolve()), "sha256": sha256_file(result_path), "evaluation_type": result.get("evaluation_type")}
    update_manifest(manifest_path, manifest, "completed", stage="pipeline", message="研究、开发验证与评估已完成；发布需要独立人工授权")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    try:
        config_path = Path(args.config).resolve()
        defaults = load_object(config_path, "pipeline defaults")
        if args.request:
            request_path = Path(args.request).resolve()
            request = deep_merge(defaults, load_object(request_path, "pipeline request"))
            request_base = request_path.parent
        else:
            request_path = config_path
            request = defaults
            research = request.setdefault("research", {})
            if not isinstance(research, dict):
                raise OrchestrationError("pipeline defaults research must be an object")
            research["idea"] = (
                args.idea.strip() if args.idea is not None
                else read_idea_file(Path(args.input).resolve())
            )
            if not research["idea"]:
                raise OrchestrationError("factor idea is empty")
            request_base = config_path.parent
        if args.factor_name:
            if not FACTOR_NAME_RE.fullmatch(args.factor_name):
                raise OrchestrationError("--factor-name must be lowercase snake_case")
            development = request.setdefault("development", {})
            if not isinstance(development, dict):
                raise OrchestrationError("effective development config must be an object")
            development["factor_name"] = args.factor_name
        manifest = run_pipeline(
            request_path, run_dir, request_override=request, request_base=request_base
        )
    except OrchestrationError as exc:
        manifest_path = run_dir / "pipeline-manifest.json"
        if manifest_path.is_file():
            manifest = load_object(manifest_path, "pipeline manifest")
            update_manifest(manifest_path, manifest, "failed", stage=manifest.get("current_stage", "pipeline"), message=str(exc))
        print(json.dumps({"success": False, "error": str(exc), "run_dir": str(run_dir)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"success": True, "status": manifest["status"], "next_action": manifest.get("next_action"), "manifest": str((run_dir / "pipeline-manifest.json").resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
