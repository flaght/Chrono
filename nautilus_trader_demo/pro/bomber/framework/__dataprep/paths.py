"""所有输入场景共用明确的配置优先级。"""
from __future__ import annotations

import os
from pathlib import Path

from .contracts import DataPaths
from .session import current_session, fail

def resolve_paths(overrides=None, env=None, required_kinds=(), project_root=None,
                  config=None, required_files=(), validate=True):
    overrides, config = dict(overrides or {}), dict(config or {})
    env = os.environ if env is None else env
    # A reusable package has no business-project root. Explicit configuration
    # wins; otherwise relative input paths follow the process working directory.
    base = Path(project_root if project_root is not None else Path.cwd()).resolve()
    sources = {}

    def choose(name, aliases=(), new_env=(), fallback=None, old_env=()):
        for values, label in ((overrides, "explicit"), (config, "config")):
            present = [(k, values[k]) for k in (name, *aliases)
                       if values.get(k) is not None and str(values[k]).strip()]
            if len({str(absolute(v)) for _, v in present}) > 1:
                fail("CONFLICTING_PATH", f"Conflicting {label} aliases for {name}")
            if present:
                sources[name] = f"{label}:{present[0][0]}"
                return present[0][1]
        for key in new_env:
            if env.get(key, "").strip():
                sources[name] = f"env:{key}"
                return env[key]
        if fallback is not None:
            sources[name] = "derived"
            return fallback
        for key in old_env:
            if env.get(key, "").strip():
                sources[name] = f"legacy_env:{key}"
                return env[key]
        return None

    def absolute(value):
        if value is None:
            return None
        candidate = Path(value).expanduser()
        return (candidate if candidate.is_absolute() else base / candidate).resolve()

    root = absolute(choose("data_root", new_env=("CTP_DATA_DIR",)))
    role = absolute(choose("role", aliases=("role_dir",), new_env=("FUT_ROLE_DATA_DIR",),
                           fallback=root / "role" if root else None, old_env=("ROLE_DIR",)))
    legacy_kline = absolute(choose("kline", aliases=("kline_dir",),
                                  new_env=("KLINE_DATA_DIR",), old_env=("KLINE_DIR",)))
    directories = {}
    for kind, dirname, env_key in (("future", "fut", "FUT_KLINE_DATA_DIR"),
                                  ("option", "opt", "OPT_KLINE_DATA_DIR"),
                                  ("index", "index", "INDEX_KLINE_DATA_DIR")):
        aliases = (dirname + "_dir", "bars_dir") if kind == "future" else (dirname + "_dir",)
        explicit_parent = legacy_kline if sources.get("kline", "").startswith(("explicit:", "config:", "env:KLINE_DATA_DIR")) else None
        fallback = (explicit_parent / dirname if explicit_parent
                    else root / "kline" / dirname if root else None)
        value = choose(dirname, aliases=aliases, new_env=(env_key,), fallback=fallback)
        supplied_asset_directory = sources.get(dirname, "").startswith(("explicit:", "config:", "env:"))
        directory = absolute(value)
        if directory is None and legacy_kline is not None:
            directory = legacy_kline / dirname if (legacy_kline / dirname).is_dir() else legacy_kline
            sources[dirname] = "legacy_kline"
        mode = overrides.get("layout", config.get("layout", "auto"))
        if mode not in {"auto", "direct", "parent"}:
            fail("INVALID_LAYOUT", "layout must be auto, direct or parent")
        if supplied_asset_directory and directory is not None and (mode == "parent" or
                mode == "auto" and (directory / dirname).is_dir()):
            directory = directory / dirname
            sources[dirname] = sources.get(dirname, "derived") + ":child"
        directories[dirname] = directory

    files = {}
    for name in ("contract_struct", "fut_basic", "opt_basic", "calendar"):
        selected = choose(name, new_env=("CTP_CALENDAR_PATH",) if name == "calendar" else ())
        if selected is None and role is not None and name != "calendar":
            if name == "contract_struct":
                canonical, legacy = role / "fut_contract.feather", role / "fut_contract_data.feather"
                if name in required_files and canonical.exists() and legacy.exists():
                    fail("CONFLICTING_PATH", "Choose --contract-struct when both role files exist", source=role)
                selected = legacy if legacy.exists() and not canonical.exists() else canonical
            else:
                selected = role / (name + ".feather")
            sources[name] = "role_default"
        files[name] = absolute(selected)

    result = DataPaths(role=role, **directories, **files, sources=sources)
    if validate:
        for kind in required_kinds:
            name = {"future": "fut", "option": "opt", "index": "index", "role": "role"}.get(kind)
            if name is None:
                fail("INVALID_ASSET", f"Unknown required input {kind}")
            directory = getattr(result, name)
            if directory is None or not directory.is_dir():
                fail("PATH_NOT_FOUND", f"Configure a valid {name} directory", source=directory)
            if name != "role" and role is not None and directory == role:
                fail("INVALID_PATH", "Reference and market directories must differ", source=directory)
        for name in required_files:
            file = getattr(result, name)
            if file is None or not file.is_file():
                fail("PATH_NOT_FOUND", f"Configure a valid {name} file", source=file)
    session = current_session()
    if session is not None:
        session.paths.append({"resolved": {k: str(v) if v is not None else None
            for k, v in {"role": role, **directories, **files}.items()}, "sources": sources})
    return result


def resolve_futures_args(args, *, require_roles=True, project_root=None):
    paths = resolve_paths(vars(args), project_root=project_root, required_kinds=("future",),
        required_files=("fut_basic", "contract_struct") if require_roles else ("fut_basic",))
    return paths


def resolve_option_args(args, *, require_futures, validate=True, project_root=None):
    kinds = ("option", "index", "future") if require_futures else ("option", "index")
    files = ("opt_basic", "fut_basic") if require_futures else ("opt_basic",)
    paths = resolve_paths(vars(args), project_root=project_root, required_kinds=kinds,
                          required_files=files, validate=validate)
    for attribute, value in (("opt_dir", paths.opt), ("index_dir", paths.index),
                             ("fut_dir", paths.fut), ("opt_basic", paths.opt_basic),
                             ("fut_basic", paths.fut_basic)):
        setattr(args, attribute, value)
    if hasattr(args, "calendar"):
        args.calendar = paths.calendar
    return args


def resolve_targets_path(path, project_root=None):
    candidate = Path(path).expanduser()
    base = Path(project_root if project_root is not None else Path.cwd())
    choices = (candidate,) if candidate.is_absolute() else (base / candidate, candidate)
    for choice in choices:
        if choice.is_file():
            return choice.resolve()
    raise FileNotFoundError(f"Target plan not found: {path}; project root={base}")
