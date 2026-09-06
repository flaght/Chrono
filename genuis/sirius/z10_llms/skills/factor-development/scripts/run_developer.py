#!/usr/bin/env python3
"""Generate one isolated Orion factor module with an environment-configured LLM."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


SKILL_DIR = Path(__file__).resolve().parent.parent
REFERENCES_DIR = SKILL_DIR / "references"
DEFAULT_FEATURE_DICTIONARY = REFERENCES_DIR / "orion_features.json"
VALIDATOR_PATH = SKILL_DIR / "scripts" / "validate_factor.py"
DEFAULT_TIMEOUT = 900
DEFAULT_MAX_RETRIES = 2
MAX_SPEC_BYTES = 1_000_000
NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
BATCH_RE = re.compile(r"^[tm][cfsb]\d{3}$")
ALLOWED_INPUT_FIELDS = {
    "trade_time", "code", "open", "high", "low", "close", "volume", "value", "openint",
    "future_open", "future_high", "future_low", "future_close", "future_volume", "future_value",
    "future_openint", "spot_open", "spot_high", "spot_low", "spot_close", "spot_volume", "spot_value",
}


class DevelopmentError(RuntimeError):
    """Expected input, configuration, provider, or generated-code error."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an isolated Orion factor module from factor-specification.json."
    )
    parser.add_argument("--spec", required=True, help="Factor specification JSON")
    parser.add_argument("--output-dir", required=True, help="New isolated artifact directory")
    parser.add_argument(
        "--feature-dictionary",
        help="Feature dictionary JSON; default: Skill built-in Orion fields",
    )
    parser.add_argument(
        "--factor-name",
        help="Optional final factor name override, for example tf001",
    )
    parser.add_argument("--provider", choices=["openai", "ollama"], help="Override provider")
    parser.add_argument("--model", help="Override model")
    parser.add_argument(
        "--env-file",
        help=".env-style configuration file (default: this Skill's .env)",
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = parser.parse_args(argv)
    if args.factor_name and not NAME_RE.fullmatch(args.factor_name):
        parser.error("--factor-name must be lowercase snake_case")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    return args


def load_env_file(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise DevelopmentError(f"env file not found: {path}")
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def setting(name: str, file_env: dict[str, str], default: str = "") -> str:
    return os.environ.get(name, file_env.get(name, default)).strip()


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DevelopmentError(f"{label} not found: {path}")
    if path.stat().st_size > MAX_SPEC_BYTES:
        raise DevelopmentError(f"{label} exceeds {MAX_SPEC_BYTES} bytes")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DevelopmentError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DevelopmentError(f"{label} must be a JSON object")
    return value


def validate_spec(spec: dict[str, Any]) -> None:
    for field in ("factor_name", "title", "summary", "formula_description"):
        if not isinstance(spec.get(field), str) or not spec[field].strip():
            raise DevelopmentError(f"factor specification {field} must be a non-empty string")
    if not NAME_RE.fullmatch(spec["factor_name"]):
        raise DevelopmentError("factor specification factor_name must be snake_case")
    required = spec.get("required_features")
    if not isinstance(required, list) or not required:
        raise DevelopmentError("factor specification required_features must be a non-empty array")
    for index, feature in enumerate(required):
        if not isinstance(feature, dict) or not isinstance(feature.get("name"), str):
            raise DevelopmentError(f"required_features[{index}].name must be a string")
    steps = spec.get("calculation_steps")
    if not isinstance(steps, list) or not steps:
        raise DevelopmentError("factor specification calculation_steps must be non-empty")


def validate_feature_dictionary(dictionary: dict[str, Any], spec: dict[str, Any]) -> None:
    features = dictionary.get("features")
    if not isinstance(features, dict) or not features:
        raise DevelopmentError("feature dictionary features must be a non-empty object")
    unregistered = sorted(set(features) - (ALLOWED_INPUT_FIELDS - {"trade_time", "code"}))
    if unregistered:
        raise DevelopmentError(
            "feature dictionary contains fields unsupported by factor-development: "
            + ", ".join(unregistered)
        )
    required = {item["name"] for item in spec["required_features"]}
    unknown = sorted(required - set(features))
    if unknown:
        raise DevelopmentError(
            "factor specification references features absent from dictionary: " + ", ".join(unknown)
        )


def extract_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    candidate = fenced.group(1) if fenced else text[start:end + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise DevelopmentError(f"LLM returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DevelopmentError("LLM JSON must be an object")
    return value


def structured_user_message(payload: dict[str, Any]) -> str:
    """把不可信开发载荷放入稳定的 XML 数据边界，同时保留 JSON 结构。"""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    escaped = serialized.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        '<development_input trust="untrusted" format="json" read_only="true">\n'
        f"{escaped}\n"
        "</development_input>"
    )


def llm_call(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    timeout: int,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": structured_user_message(user_payload)},
    ]
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if provider == "openai":
        url = f"{base_url.rstrip('/')}/chat/completions"
        body = {"model": model, "messages": messages, "temperature": 0.1, "stream": True}
    else:
        url = f"{base_url.rstrip('/')}/api/chat"
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.1},
        }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    chunks: list[str] = []
    non_sse_lines: list[str] = []
    print("\n[因子代码生成] 模型开始流式生成：", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                text: Any = None
                if provider == "openai":
                    if line == "data: [DONE]":
                        break
                    if not line.startswith("data:"):
                        non_sse_lines.append(line)
                        continue
                    encoded = line[5:].strip()
                    if not encoded or encoded == "[DONE]":
                        continue
                    try:
                        event = json.loads(encoded)
                    except json.JSONDecodeError as exc:
                        raise DevelopmentError(f"invalid OpenAI stream event: {exc}") from exc
                    if isinstance(event, dict) and event.get("error"):
                        raise DevelopmentError(f"OpenAI-compatible stream error: {event['error']}")
                    choices = event.get("choices") if isinstance(event, dict) else None
                    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                        continue
                    delta = choices[0].get("delta")
                    if isinstance(delta, dict):
                        text = delta.get("content")
                    if text is None and not chunks:
                        message = choices[0].get("message")
                        text = message.get("content") if isinstance(message, dict) else None
                else:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise DevelopmentError(f"invalid Ollama stream event: {exc}") from exc
                    if isinstance(event, dict) and event.get("error"):
                        raise DevelopmentError(f"Ollama stream error: {event['error']}")
                    message = event.get("message") if isinstance(event, dict) else None
                    text = message.get("content") if isinstance(message, dict) else None
                if isinstance(text, str) and text:
                    chunks.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise DevelopmentError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DevelopmentError(f"network error calling {url}: {exc.reason}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise DevelopmentError(
            f"generation stream timed out after {timeout}s; retry or increase --timeout"
        ) from exc
    finally:
        print(flush=True)
    if chunks:
        return "".join(chunks)
    if provider == "openai" and non_sse_lines:
        try:
            response = json.loads("".join(non_sse_lines))
            text = response["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise DevelopmentError("invalid non-stream OpenAI-compatible response") from exc
        if isinstance(text, str) and text:
            print(text, flush=True)
            return text
    raise DevelopmentError("generation stream completed without text content")


def load_validator() -> Callable[[Path], list[str]]:
    module_spec = importlib.util.spec_from_file_location("factor_development_validator", VALIDATOR_PATH)
    if module_spec is None or module_spec.loader is None:
        raise DevelopmentError("unable to load factor validator")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.validate


def validate_generation(
    value: dict[str, Any],
    factor_spec: dict[str, Any],
    validate_file: Callable[[Path], list[str]],
    allowed_input_fields: set[str] | None = None,
    requested_factor_name: str | None = None,
) -> dict[str, Any]:
    effective_allowed = allowed_input_fields or ALLOWED_INPUT_FIELDS
    factor_name = requested_factor_name or factor_spec["factor_name"]
    if value.get("factor_name") != factor_name:
        raise DevelopmentError(f"generated factor_name must equal {factor_name!r}")
    batch = value.get("batch")
    if not isinstance(batch, str) or not BATCH_RE.fullmatch(batch):
        raise DevelopmentError("generated batch is invalid")
    if value.get("file_name") != f"{factor_name}.py":
        raise DevelopmentError(f"generated file_name must equal {factor_name}.py")
    if not isinstance(value.get("max_window"), int) or value["max_window"] < 1:
        raise DevelopmentError("generated max_window must be a positive integer")
    fields = value.get("required_input_fields")
    if not isinstance(fields, list) or any(not isinstance(item, str) for item in fields):
        raise DevelopmentError("generated required_input_fields must be a string array")
    unknown = sorted(set(fields) - effective_allowed)
    if unknown:
        raise DevelopmentError("generated code declares unregistered input fields: " + ", ".join(unknown))
    code = value.get("code")
    if not isinstance(code, str) or not code.strip():
        raise DevelopmentError("generated code must be a non-empty string")
    if "```" in code:
        raise DevelopmentError("generated code must not contain Markdown fences")
    try:
        code_tree = ast.parse(code, filename=f"{factor_name}.py")
        compile(code, f"{factor_name}.py", "exec")
    except SyntaxError as exc:
        raise DevelopmentError(f"generated Python does not compile: {exc}") from exc
    with tempfile.TemporaryDirectory(prefix="factor-development-") as tmp:
        path = Path(tmp) / batch / f"{factor_name}.py"
        path.parent.mkdir(parents=True)
        path.write_text(code, encoding="utf-8")
        errors = validate_file(path)
    if errors:
        raise DevelopmentError("factor contract validation failed: " + "; ".join(errors))
    string_constants = {
        node.targets[0].id: node.value.value
        for node in code_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }

    def resolve_string(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return string_constants.get(node.id)
        return None

    calls = [node for node in ast.walk(code_tree) if isinstance(node, ast.Call)]
    expected_output = ["trade_time", "code", factor_name]
    has_exact_output = any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "select"
        and len(call.args) == 1
        and isinstance(call.args[0], (ast.List, ast.Tuple))
        and [resolve_string(item) for item in call.args[0].elts] == expected_output
        for call in calls
    )
    has_exact_alias = any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "alias"
        and call.args
        and resolve_string(call.args[0]) == factor_name
        for call in calls
    )
    if not has_exact_output or not has_exact_alias:
        raise DevelopmentError(
            "generated single-factor code must alias and select exactly "
            f"{expected_output!r}"
        )
    referenced_base_fields = {
        call.args[0].value
        for call in ast.walk(code_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "pl"
        and call.func.attr == "col"
        and call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
        and call.args[0].value in ALLOWED_INPUT_FIELDS
    }
    expected_fields = {"trade_time", "code", *referenced_base_fields}
    declared_fields = set(fields)
    if declared_fields != expected_fields:
        missing = sorted(expected_fields - declared_fields)
        extra = sorted(declared_fields - expected_fields)
        raise DevelopmentError(
            "required_input_fields must exactly match code inputs; "
            f"missing={missing}, extra={extra}"
        )
    notes = value.get("implementation_notes")
    if not isinstance(notes, list) or any(not isinstance(item, str) for item in notes):
        raise DevelopmentError("generated implementation_notes must be a string array")
    return value


def request_validated(
    call: Callable[[dict[str, Any]], str],
    validate: Callable[[dict[str, Any]], dict[str, Any]],
    payload: dict[str, Any],
    max_retries: int,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    current = dict(payload)
    last_raw = ""
    for attempt in range(1, max_retries + 2):
        started = time.monotonic()
        print(f"[factor-development] 第 {attempt}/{max_retries + 1} 次请求", flush=True)
        try:
            last_raw = call(current)
            result = validate(extract_json(last_raw))
            attempts.append({"attempt": attempt, "status": "passed", "seconds": round(time.monotonic() - started, 3)})
            return result, last_raw, attempts
        except DevelopmentError as exc:
            attempts.append({
                "attempt": attempt,
                "status": "failed",
                "seconds": round(time.monotonic() - started, 3),
                "error": str(exc),
            })
            if attempt > max_retries:
                raise DevelopmentError(f"generation failed after {attempt} attempt(s): {exc}") from exc
            print(f"[factor-development] 响应无效，准备重试：{exc}", file=sys.stderr, flush=True)
            current = dict(payload)
            current["previous_response_error"] = str(exc)
            current["retry_instruction"] = (
                "返回修正后的完整 JSON；不要解释，不要使用 Markdown 围栏。"
                "先逐项修复 previous_response_error，再依据 reference_code_patterns 做生成前自检。"
                "类型必须写在函数签名中；compute docstring 必须逐字包含 df_lazy、"
                "pl.LazyFrame、trade_time、code、完整 factor_name 和全部基础输入字段。"
            )
    raise AssertionError("unreachable")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise DevelopmentError(f"output directory already exists; choose a new path: {out_dir}")
    factor_spec = load_json_object(Path(args.spec), "factor specification")
    validate_spec(factor_spec)
    feature_dictionary_path = (
        Path(args.feature_dictionary) if args.feature_dictionary
        else DEFAULT_FEATURE_DICTIONARY
    )
    feature_dictionary = load_json_object(feature_dictionary_path, "feature dictionary")
    validate_feature_dictionary(feature_dictionary, factor_spec)
    effective_allowed_fields = {
        "trade_time", "code", *feature_dictionary["features"].keys()
    }
    feature_dictionary_origin = "provided" if args.feature_dictionary else "skill_default"

    env_path = Path(args.env_file) if args.env_file else SKILL_DIR / ".env"
    file_env = load_env_file(env_path if env_path.is_file() or args.env_file else None)
    provider = args.provider or setting("FACTOR_LLM_PROVIDER", file_env, "openai")
    if provider not in {"openai", "ollama"}:
        raise DevelopmentError("FACTOR_LLM_PROVIDER must be openai or ollama")
    default_model = "gpt-4.1-mini" if provider == "openai" else "qwen3:8b"
    model = args.model or setting("FACTOR_LLM_MODEL", file_env, default_model)
    if provider == "openai":
        base_url = setting("OPENAI_BASE_URL", file_env, "https://api.openai.com/v1")
        api_key = setting("OPENAI_API_KEY", file_env)
        if not api_key:
            raise DevelopmentError("OPENAI_API_KEY is required for provider openai")
    else:
        base_url = setting("OLLAMA_BASE_URL", file_env, "http://localhost:11434")
        api_key = setting("OLLAMA_API_KEY", file_env)

    prompt = (REFERENCES_DIR / "developer_prompt.md").read_text(encoding="utf-8")
    contract = (REFERENCES_DIR / "factor-contract.md").read_text(encoding="utf-8")
    patterns = (REFERENCES_DIR / "generation-patterns.md").read_text(encoding="utf-8")
    payload = {
        "factor_specification_untrusted": factor_spec,
        "feature_dictionary_untrusted": feature_dictionary,
        "requested_factor_name": args.factor_name,
        "allowed_input_fields": sorted(effective_allowed_fields),
        "factor_contract": contract,
        "reference_code_patterns": patterns,
    }
    validate_file = load_validator()
    generated, raw, attempts = request_validated(
        lambda current: llm_call(provider, model, base_url, api_key, prompt, current, args.timeout),
        lambda value: validate_generation(
            value, factor_spec, validate_file, effective_allowed_fields, args.factor_name
        ),
        payload,
        args.max_retries,
    )

    factor_name = generated["factor_name"]
    batch = generated["batch"]
    factor_rel = Path("candidate") / "feature" / batch / f"{factor_name}.py"
    out_dir.mkdir(parents=True, exist_ok=False)
    factor_path = out_dir / factor_rel
    factor_path.parent.mkdir(parents=True)
    factor_path.write_text(generated["code"], encoding="utf-8")
    (out_dir / "generation-response.raw.txt").write_text(raw, encoding="utf-8")

    request_record = {
        "spec_source": str(Path(args.spec).resolve()),
        "feature_dictionary_source": str(feature_dictionary_path.resolve()),
        "feature_dictionary_origin": feature_dictionary_origin,
        "requested_factor_name": args.factor_name,
        "spec_factor_name": factor_spec["factor_name"],
        "provider": provider,
        "model": model,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
    }
    generation_report = {
        "success": True,
        "factor_name": factor_name,
        "batch": batch,
        "max_window": generated["max_window"],
        "required_input_fields": generated["required_input_fields"],
        "implementation_notes": generated["implementation_notes"],
        "attempts": attempts,
    }
    validation_report = {
        "status": "awaiting_runtime_validation",
        "ready_for_evaluation": False,
        "checks": {
            "response_contract": "passed",
            "python_ast_and_compile": "passed",
            "factor_static_contract": "passed",
            "human_review": "optional",
            "runtime_lazyframe": "pending",
        },
    }
    artifacts = {
        "request.json": request_record,
        "generation-report.json": generation_report,
        "validation-report.json": validation_report,
    }
    for name, value in artifacts.items():
        (out_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "awaiting_runtime_validation",
        "ready_for_evaluation": False,
        "factor_file": str(factor_rel),
        "factor_sha256": sha256_text(generated["code"]),
        "artifacts": [str(factor_rel), *artifacts.keys(), "generation-response.raw.txt", "manifest.json"],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "success": True,
        "status": "awaiting_runtime_validation",
        "ready_for_evaluation": False,
        "factor_file": str(factor_path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DevelopmentError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
