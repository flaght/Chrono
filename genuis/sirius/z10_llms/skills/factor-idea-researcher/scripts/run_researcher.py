#!/usr/bin/env python3
"""Single-pass factor idea researcher with direct, search, and fetch modes."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import ipaddress
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


SKILL_DIR = Path(__file__).resolve().parent.parent
REFERENCES_DIR = SKILL_DIR / "references"
DEFAULT_FEATURE_DICTIONARY = REFERENCES_DIR / "orion_features.json"
DEFAULT_TIMEOUT = 900
DEFAULT_MAX_RETRIES = 2
MAX_IDEA_CHARS = 20_000
MAX_FEATURES = 500
MAX_FETCH_CHARS = 20_000
MAX_TOTAL_CONTEXT_CHARS = 60_000


class ResearchError(RuntimeError):
    """Expected user/config/provider error."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._ignored:
            self._ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored and data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Turn one factor idea into a complete factor specification."
    )
    idea = parser.add_mutually_exclusive_group(required=True)
    idea.add_argument("--idea", help="Factor idea as text")
    idea.add_argument("--input", help="UTF-8 text/Markdown file containing the factor idea")
    parser.add_argument("--mode", required=True, choices=["direct", "search", "fetch"])
    parser.add_argument(
        "--feature-dictionary",
        help="Feature dictionary JSON; default: Skill built-in Orion fields",
    )
    parser.add_argument("--query", help="Search query override for search mode")
    parser.add_argument("--url", action="append", default=[], help="URL to fetch; repeatable")
    parser.add_argument("--max-results", type=int, default=5, help="Search results, 1-10")
    parser.add_argument("--provider", choices=["openai", "ollama"], help="Override provider")
    parser.add_argument("--model", help="Override model")
    parser.add_argument(
        "--env-file",
        help=".env-style configuration file (default: this Skill's .env)",
    )
    parser.add_argument("--output-dir", help="Artifact directory")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = parser.parse_args(argv)

    if args.mode == "fetch" and not args.url:
        parser.error("--mode fetch requires at least one --url")
    if args.mode != "fetch" and args.url:
        parser.error("--url is only valid with --mode fetch")
    if not 1 <= args.max_results <= 10:
        parser.error("--max-results must be between 1 and 10")
    if args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    return args


def load_env_file(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise ResearchError(f"env file not found: {path}")
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


def read_idea(args: argparse.Namespace) -> str:
    if args.idea is not None:
        idea = args.idea.strip()
    else:
        path = Path(args.input)
        if not path.is_file():
            raise ResearchError(f"idea input not found: {path}")
        if path.suffix.lower() not in {".txt", ".md", ".markdown"}:
            raise ResearchError("idea input must be a .txt, .md, or .markdown file")
        try:
            idea = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ResearchError(f"failed to read UTF-8 idea input: {exc}") from exc
    if not idea:
        raise ResearchError("factor idea must not be empty")
    if len(idea) > MAX_IDEA_CHARS:
        raise ResearchError(f"factor idea exceeds {MAX_IDEA_CHARS} characters")
    return idea


def load_feature_dictionary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResearchError(f"feature dictionary not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResearchError(f"invalid feature dictionary JSON: {exc}") from exc
    errors: list[str] = []
    if not isinstance(data, dict):
        raise ResearchError("feature dictionary must be a JSON object")
    for key in ("id", "version", "timeframe"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            errors.append(f"{key} must be a non-empty string")
    features = data.get("features")
    if not isinstance(features, dict) or not features:
        errors.append("features must be a non-empty object")
    elif len(features) > MAX_FEATURES:
        errors.append(f"features exceeds maximum {MAX_FEATURES}")
    else:
        for name, spec in features.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                errors.append(f"invalid feature name: {name!r}")
                continue
            if not isinstance(spec, dict):
                errors.append(f"feature {name} must be an object")
                continue
            for field in ("dtype", "description", "unit", "source", "availability"):
                if not isinstance(spec.get(field), str) or not spec[field].strip():
                    errors.append(f"feature {name}.{field} must be a non-empty string")
            if not isinstance(spec.get("nullable"), bool):
                errors.append(f"feature {name}.nullable must be boolean")
    if errors:
        raise ResearchError("feature dictionary validation failed: " + "; ".join(errors[:20]))
    return data


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ResearchError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ResearchError(f"network error calling {url}: {exc.reason}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise ResearchError(
            f"request to {url} timed out after {timeout}s; retry or increase --timeout"
        ) from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ResearchError(f"provider returned invalid JSON from {url}") from exc
    if not isinstance(parsed, dict):
        raise ResearchError(f"provider returned non-object JSON from {url}")
    return parsed


def search_tavily(query: str, api_key: str, base_url: str, max_results: int, timeout: int) -> list[dict[str, str]]:
    if not api_key:
        raise ResearchError("TAVILY_API_KEY is required for search mode")
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
    }
    response = post_json(f"{base_url.rstrip('/')}/search", payload, {}, timeout)
    results: list[dict[str, str]] = []
    for item in response.get("results", []):
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        results.append({
            "title": str(item.get("title", ""))[:500],
            "url": url,
            "content": str(item.get("content", ""))[:4000],
        })
    if not results:
        raise ResearchError("search returned no usable results")
    return results


def validate_public_url(raw_url: str) -> str:
    parsed = urllib.parse.urlparse(raw_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ResearchError(f"URL must be public HTTP(S): {raw_url}")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        raise ResearchError(f"local URL is not allowed: {raw_url}")
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        ip = None
    if ip and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved):
        raise ResearchError(f"private or reserved URL is not allowed: {raw_url}")
    return raw_url


def get_text(url: str, timeout: int) -> tuple[str, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "KD-HYS-FactorResearcher/0.1", "Accept": "text/plain,text/html"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_FETCH_CHARS * 4)
            content_type = response.headers.get("Content-Type", "")
    except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise ResearchError(f"failed to fetch {url}: {exc}") from exc
    text = raw.decode("utf-8", errors="replace")
    if "html" in content_type.lower() or "<html" in text[:500].lower():
        parser = _TextExtractor()
        parser.feed(text)
        text = parser.text()
    return text.strip()[:MAX_FETCH_CHARS], content_type


def fetch_sources(urls: list[str], reader_base: str, timeout: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    failures: list[str] = []
    for raw_url in urls:
        url = validate_public_url(raw_url)
        reader_url = f"{reader_base.rstrip('/')}/{url}"
        try:
            content, _ = get_text(reader_url, timeout)
            method = "jina_reader"
        except ResearchError:
            try:
                content, _ = get_text(url, timeout)
                method = "direct_http"
            except ResearchError as exc:
                failures.append(str(exc))
                continue
        if content:
            results.append({"url": url, "method": method, "content": content})
    if not results:
        raise ResearchError("all fetches failed: " + "; ".join(failures[:5]))
    return results


def extract_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ResearchError(f"LLM returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ResearchError("LLM JSON must be an object")
    return value


def structured_user_message(payload: dict[str, Any]) -> str:
    """把不可信载荷放入稳定的 XML 数据边界，同时保留 JSON 结构。"""
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    escaped = serialized.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        '<research_input trust="untrusted" format="json" read_only="true">\n'
        f"{escaped}\n"
        "</research_input>"
    )


def llm_call(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    timeout: int,
    stage: str,
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
        body = {"model": model, "messages": messages, "temperature": 0.2, "stream": True}
    else:
        url = f"{base_url.rstrip('/')}/api/chat"
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.2},
        }

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    chunks: list[str] = []
    non_sse_lines: list[str] = []
    print(f"\n[{stage}] 模型开始流式生成：", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
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
                        raise ResearchError(f"invalid OpenAI stream event: {exc}") from exc
                    if isinstance(event, dict) and event.get("error"):
                        raise ResearchError(f"OpenAI-compatible stream error: {event['error']}")
                    if not isinstance(event, dict):
                        raise ResearchError("OpenAI-compatible stream event must be an object")
                    choices = event.get("choices")
                    # Usage/statistics events and some provider-specific terminal events
                    # legitimately contain no choices. They do not invalidate collected text.
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if isinstance(delta, dict):
                        text = delta.get("content")
                    else:
                        text = None
                    # A few compatible gateways return one full message inside a stream.
                    # Only accept it before any delta text to avoid duplicating content.
                    if text is None and not chunks:
                        message = choice.get("message")
                        text = message.get("content") if isinstance(message, dict) else None
                else:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ResearchError(f"invalid Ollama stream event: {exc}") from exc
                    if isinstance(event, dict) and event.get("error"):
                        raise ResearchError(f"Ollama stream error: {event['error']}")
                    message = event.get("message", {}) if isinstance(event, dict) else {}
                    text = message.get("content") if isinstance(message, dict) else None
                if isinstance(text, str) and text:
                    chunks.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ResearchError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ResearchError(f"network error calling {url}: {exc.reason}") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise ResearchError(
            f"{stage} stream timed out after {timeout}s; retry or increase --timeout"
        ) from exc
    finally:
        print(flush=True)

    if chunks:
        return "".join(chunks)

    # Some OpenAI-compatible gateways ignore stream=true and return one normal JSON body.
    if provider == "openai" and non_sse_lines:
        try:
            response = json.loads("".join(non_sse_lines))
            text = response["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ResearchError("invalid non-stream fallback response from OpenAI-compatible provider") from exc
        if isinstance(text, str) and text:
            sys.stdout.write(text)
            sys.stdout.flush()
            print(flush=True)
            return text
    raise ResearchError(f"{stage} stream completed without text content")


def string_list(value: Any, field: str, minimum: int = 0) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise ResearchError(f"{field} must be an array with at least {minimum} item(s)")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ResearchError(f"{field} must contain only non-empty strings")
    return value


def validate_first_principles(data: dict[str, Any], feature_names: set[str] | None) -> None:
    for field in ("problem_definition", "reconstructed_hypothesis"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ResearchError(f"first-principles {field} must be a non-empty string")
    for field, minimum in (
        ("surface_assumptions", 1), ("fundamental_truths", 1), ("hard_constraints", 1),
        ("testable_mechanisms", 1), ("available_features", 0), ("missing_features", 0),
        ("falsification_conditions", 1),
    ):
        string_list(data.get(field), f"first-principles {field}", minimum)
    unknown_available = set(data["available_features"]) - feature_names if feature_names is not None else set()
    if feature_names is not None and unknown_available:
        raise ResearchError(
            "first-principles available_features contains dictionary-external names: "
            + ", ".join(sorted(unknown_available))
        )


def validate_factor_spec(
    data: dict[str, Any], feature_names: set[str] | None, timeframe: str | None = None
) -> None:
    for field in ("factor_name", "title", "summary", "hypothesis", "economic_rationale", "formula_description"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ResearchError(f"factor specification {field} must be a non-empty string")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", data["factor_name"]):
        raise ResearchError("factor_name must be snake_case")
    required = data.get("required_features")
    if not isinstance(required, list) or not required:
        raise ResearchError("required_features must be a non-empty array")
    used: set[str] = set()
    for index, item in enumerate(required):
        if not isinstance(item, dict):
            raise ResearchError(f"required_features[{index}] must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ResearchError(f"required_features[{index}].name is invalid: {name!r}")
        if feature_names is not None and name not in feature_names:
            raise ResearchError(f"required_features[{index}].name is not in feature dictionary: {name!r}")
        used.add(name)
        if not isinstance(item.get("role"), str) or not item["role"].strip():
            raise ResearchError(f"required_features[{index}].role must be non-empty")
        string_list(item.get("transformations"), f"required_features[{index}].transformations")
    for field, minimum in (
        ("missing_features", 0), ("calculation_steps", 1), ("expected_behavior", 1),
        ("falsification_conditions", 1), ("validation_plan", 1), ("risk_warnings", 1),
    ):
        string_list(data.get(field), field, minimum)
    parameters = data.get("parameters")
    if not isinstance(parameters, list):
        raise ResearchError("parameters must be an array")
    for index, item in enumerate(parameters):
        if not isinstance(item, dict):
            raise ResearchError(f"parameters[{index}] must be an object")
        for field in ("name", "type", "meaning"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ResearchError(f"parameters[{index}].{field} must be non-empty")
        if item["type"] not in {"integer", "number", "string", "boolean"}:
            raise ResearchError(f"parameters[{index}].type is unsupported: {item['type']!r}")
        if "default" not in item:
            raise ResearchError(f"parameters[{index}].default is required")
    requirements = data.get("data_requirements")
    if not isinstance(requirements, dict):
        raise ResearchError("data_requirements must be an object")
    for field in ("timeframe", "alignment", "missing_value_policy"):
        if not isinstance(requirements.get(field), str) or not requirements[field].strip():
            raise ResearchError(f"data_requirements.{field} must be non-empty")
    if not isinstance(requirements.get("minimum_history_bars"), int) or requirements["minimum_history_bars"] < 1:
        raise ResearchError("data_requirements.minimum_history_bars must be a positive integer")
    if timeframe and requirements["timeframe"] != timeframe:
        raise ResearchError(
            f"data_requirements.timeframe must match feature dictionary {timeframe!r}"
        )
    evidence = data.get("evidence")
    if not isinstance(evidence, list):
        raise ResearchError("evidence must be an array")
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise ResearchError(f"evidence[{index}] must be an object")
        for field in ("source_ref", "claim"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ResearchError(f"evidence[{index}].{field} must be non-empty")


def request_validated(
    call,
    validator,
    user_payload: dict[str, Any],
    max_retries: int,
    stage: str,
) -> tuple[dict[str, Any], str, list[str]]:
    attempts: list[str] = []
    payload = dict(user_payload)
    last_raw = ""
    for attempt in range(1, max_retries + 2):
        started = time.monotonic()
        print(f"[{stage}] 第 {attempt}/{max_retries + 1} 次请求", flush=True)
        try:
            last_raw = call(payload)
            parsed = extract_json(last_raw)
            validator(parsed)
            attempts.append(f"attempt {attempt}: passed in {time.monotonic() - started:.2f}s")
            return parsed, last_raw, attempts
        except ResearchError as exc:
            attempts.append(f"attempt {attempt}: {exc}")
            if attempt > max_retries:
                raise ResearchError(f"{stage} failed after {attempt} attempt(s): {exc}") from exc
            print(f"[{stage}] 响应无效，准备重试：{exc}", file=sys.stderr, flush=True)
            payload = dict(user_payload)
            payload["previous_response_error"] = str(exc)
            payload["retry_instruction"] = "Return a complete corrected JSON object matching the system schema."
    raise AssertionError("unreachable")


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env_path = Path(args.env_file) if args.env_file else SKILL_DIR / ".env"
    file_env = load_env_file(env_path if env_path.is_file() or args.env_file else None)
    idea = read_idea(args)
    feature_dictionary_path = (
        Path(args.feature_dictionary) if args.feature_dictionary
        else DEFAULT_FEATURE_DICTIONARY
    )
    feature_dictionary = load_feature_dictionary(feature_dictionary_path)
    feature_names = set(feature_dictionary["features"])
    feature_mode = "constrained"
    feature_dictionary_origin = "provided" if args.feature_dictionary else "skill_default"

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / "factor-idea-runs" / timestamp
    if out_dir.exists():
        raise ResearchError(f"output directory already exists; choose a new path: {out_dir}")

    provider = args.provider or setting("FACTOR_LLM_PROVIDER", file_env, "openai")
    if provider not in {"openai", "ollama"}:
        raise ResearchError("FACTOR_LLM_PROVIDER must be openai or ollama")
    default_model = "gpt-4.1-mini" if provider == "openai" else "qwen3:8b"
    model = args.model or setting("FACTOR_LLM_MODEL", file_env, default_model)
    if provider == "openai":
        base_url = setting("OPENAI_BASE_URL", file_env, "https://api.openai.com/v1")
        api_key = setting("OPENAI_API_KEY", file_env)
        if not api_key:
            raise ResearchError("OPENAI_API_KEY is required for provider openai")
    else:
        base_url = setting("OLLAMA_BASE_URL", file_env, "http://localhost:11434")
        api_key = setting("OLLAMA_API_KEY", file_env)

    if args.mode == "search":
        query = (args.query or idea).strip()
        research_context: dict[str, Any] = {
            "mode": "search",
            "query": query,
            "results": search_tavily(
                query,
                setting("TAVILY_API_KEY", file_env),
                setting("TAVILY_BASE_URL", file_env, "https://api.tavily.com"),
                args.max_results,
                args.timeout,
            ),
        }
    elif args.mode == "fetch":
        research_context = {
            "mode": "fetch",
            "documents": fetch_sources(
                args.url,
                setting("JINA_READER_BASE_URL", file_env, "https://r.jina.ai/"),
                args.timeout,
            ),
        }
    else:
        research_context = {"mode": "direct", "evidence": []}

    context_json = json.dumps(research_context, ensure_ascii=False)
    if len(context_json) > MAX_TOTAL_CONTEXT_CHARS:
        raise ResearchError(f"research context exceeds {MAX_TOTAL_CONTEXT_CHARS} characters")

    first_prompt = (REFERENCES_DIR / "first_principles_prompt.md").read_text(encoding="utf-8")
    spec_prompt = (REFERENCES_DIR / "factor_spec_prompt.md").read_text(encoding="utf-8")

    def call_with(prompt: str, payload: dict[str, Any], stage: str) -> str:
        return llm_call(
            provider, model, base_url, api_key, prompt, payload, args.timeout, stage
        )

    first_payload = {
        "factor_idea_untrusted": idea,
        "feature_mode": feature_mode,
        "feature_dictionary": feature_dictionary,
        "research_context_untrusted": research_context,
    }
    first, first_raw, first_attempts = request_validated(
        lambda payload: call_with(first_prompt, payload, "第一性原理分析"),
        lambda data: validate_first_principles(data, feature_names),
        first_payload,
        args.max_retries,
        "first-principles",
    )

    spec_payload = {
        "factor_idea_untrusted": idea,
        "feature_mode": feature_mode,
        "feature_dictionary": feature_dictionary,
        "research_context_untrusted": research_context,
        "first_principles_analysis": first,
    }
    spec, spec_raw, spec_attempts = request_validated(
        lambda payload: call_with(spec_prompt, payload, "因子说明书生成"),
        lambda data: validate_factor_spec(
            data,
            feature_names,
            feature_dictionary["timeframe"] if args.feature_dictionary else None,
        ),
        spec_payload,
        args.max_retries,
        "factor-specification",
    )

    out_dir.mkdir(parents=True, exist_ok=False)

    request_record = {
        "idea": idea,
        "mode": args.mode,
        "feature_mode": feature_mode,
        "query": args.query,
        "urls": args.url,
        "feature_dictionary_source": str(feature_dictionary_path.resolve()),
        "feature_dictionary_origin": feature_dictionary_origin,
        "feature_dictionary_id": feature_dictionary["id"],
        "feature_dictionary_version": feature_dictionary["version"],
        "provider": provider,
        "model": model,
    }
    artifacts: dict[str, Any] = {
        "request.json": request_record,
        "research-context.json": research_context,
        "first-principles.json": first,
        "factor-specification.json": spec,
        "validation-report.json": {
            "first_principles": first_attempts,
            "factor_specification": spec_attempts,
        },
    }
    artifacts["feature-dictionary.json"] = feature_dictionary
    for filename, value in artifacts.items():
        (out_dir / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "first-principles.raw.txt").write_text(first_raw, encoding="utf-8")
    (out_dir / "factor-specification.raw.txt").write_text(spec_raw, encoding="utf-8")

    manifest = {
        "schema_version": "1.0.0",
        "run_id": timestamp,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "completed",
        "mode": args.mode,
        "feature_mode": feature_mode,
        "feature_dictionary_origin": feature_dictionary_origin,
        "provider": provider,
        "model": model,
        "feature_dictionary": {
            "id": feature_dictionary["id"],
            "version": feature_dictionary["version"],
            "sha256": sha256_json(feature_dictionary),
            "source": str(feature_dictionary_path.resolve()),
            "origin": feature_dictionary_origin,
        },
        "artifacts": [*artifacts.keys(), "first-principles.raw.txt", "factor-specification.raw.txt"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "success": True,
        "output_dir": str(out_dir.resolve()),
        "factor_name": spec["factor_name"],
        "missing_features": spec["missing_features"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResearchError as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
