#!/usr/bin/env python3
"""同步 bomber/framework 源码，开发与发布保持同一导入路径。

默认只报告差异；--apply 构建 <pro>/dist/bomber/framework。
--dist-root 可指定目标项目根目录。已有目录须有同步标记，或显式传入
--adopt-existing 接管旧的 framework 副本。不会同步 pro/bomber/__init__.py。
"""
from __future__ import annotations

import argparse
import ast
import difflib
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "bomber" / "framework"
PACKAGE_NAMES = {"datahub", "dataprep", "market", "trader"}
LEGACY_IMPORT_ROOTS = PACKAGE_NAMES | {"ba", "fixes"}
EXCLUDED_DIRS = {".git", ".build-deps", "__pycache__", "build", "dist", ".venv",
                 ".pytest_cache", ".mypy_cache", ".ruff_cache"}
MARKER = ".sync-bomber-framework"


def old_imports(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level and node.module:
            if node.module.split(".", 1)[0] in LEGACY_IMPORT_ROOTS:
                found.append(f"line {node.lineno}: from {node.module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in LEGACY_IMPORT_ROOTS:
                    found.append(f"line {node.lineno}: import {alias.name}")
    return found


def excluded(relative: Path) -> bool:
    if EXCLUDED_DIRS.intersection(relative.parts) or relative.name in {MARKER, ".DS_Store"}:
        return True
    if relative.suffix in {".pyc", ".pyo", ".o", ".obj"}:
        return True
    # Ship Cython sources; rebuild extension binaries for the target engine/ABI.
    # CTP vendor shared libraries in binding/ are source-build inputs and remain.
    if relative.parent == Path("market/basic"):
        if relative.suffix in {".so", ".pyd", ".dll"}:
            return True
        if relative.name in {"custom_bar.c", "fast_factory.c", "custom_bar.html", "fast_factory.html"}:
            return True
    return False


def staged_files(source_root: Path) -> dict[Path, bytes]:
    for package in sorted(PACKAGE_NAMES):
        if not (source_root / package).is_dir():
            raise ValueError(f"missing source package: {source_root / package}")
    for name in ("__init__.py", "fixes.py"):
        if not (source_root / name).is_file():
            raise ValueError(f"missing source file: {source_root / name}")
    files = {}
    for source in sorted(source_root.rglob("*")):
        relative = source.relative_to(source_root)
        if excluded(relative):
            continue
        if source.is_symlink():
            raise ValueError(f"source symlinks are not supported: {source}")
        if not source.is_file():
            continue
        content = source.read_bytes()
        if source.suffix == ".py":
            remaining = old_imports(ast.parse(content, filename=str(source)))
            if remaining:
                raise ValueError(f"legacy imports in {source}: {remaining}; fix the source before publishing")
        files[relative] = content
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="framework directory, or a project containing bomber/framework")
    parser.add_argument("--dist-root", type=Path, default=PROJECT_ROOT / "dist",
                        help="target project root; defaults to <pro>/dist")
    parser.add_argument("--apply", action="store_true", help="replace the staged framework copy")
    parser.add_argument("--diff", action="store_true", help="show diffs for text files")
    parser.add_argument("--adopt-existing", action="store_true",
                        help="explicitly allow replacing an existing unmarked framework directory")
    args = parser.parse_args()
    source_root = args.source.resolve()
    if (source_root / "bomber" / "framework").is_dir():
        source_root = source_root / "bomber" / "framework"
    if not source_root.is_dir():
        parser.error(f"source directory does not exist: {source_root}")
    destination = args.dist_root.resolve() / "bomber" / "framework"
    if destination.is_symlink():
        parser.error(f"destination must not be a symlink: {destination}")
    resolved_destination = destination.resolve()
    if (resolved_destination == source_root or source_root in resolved_destination.parents
            or resolved_destination in source_root.parents):
        parser.error("source and destination must be separate, non-overlapping directories")
    try:
        files = staged_files(source_root)
        if destination.exists():
            if not destination.is_dir():
                raise ValueError(f"destination is not a directory: {destination}")
            if not (destination / MARKER).is_file() and not args.adopt_existing:
                raise ValueError(f"refusing to replace unmarked directory: {destination}; "
                                 "use --adopt-existing to explicitly adopt this framework copy")
            if any(p.is_symlink() for p in destination.rglob("*")):
                raise ValueError(f"destination contains symlinks: {destination}")
        existing = {p.relative_to(destination): p.read_bytes()
                    for p in destination.rglob("*") if p.is_file() and p.name != MARKER}
    except (OSError, SyntaxError, ValueError) as exc:
        parser.error(str(exc))
    changed = [p for p, content in sorted(files.items()) if existing.get(p) != content]
    stale = sorted(existing.keys() - files.keys())
    for relative in changed:
        print(f"{'STAGE' if args.apply else 'DRIFT'} {relative}")
        if args.diff:
            try:
                before = existing.get(relative, b"").decode("utf-8")
                after = files[relative].decode("utf-8")
            except UnicodeDecodeError:
                print(f"BINARY {relative}")
            else:
                sys.stdout.writelines(difflib.unified_diff(
                    before.splitlines(keepends=True), after.splitlines(keepends=True),
                    fromfile=str(destination / relative), tofile=str(source_root / relative)))
    for relative in stale:
        print(f"{'REMOVE' if args.apply else 'STALE'} {relative}")
    if args.apply:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".framework-stage-", dir=destination.parent) as temporary:
            staged = Path(temporary) / "framework"
            staged.mkdir()
            for relative in files:
                output = staged / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_root / relative, output)
            (staged / MARKER).write_text("Generated by sync-bomber-framework.py; edit the source in pro.\n")
            # Retain the previous copy until the new directory is in place.
            previous = Path(temporary) / "previous"
            if destination.exists():
                destination.rename(previous)
            try:
                staged.rename(destination)
            except OSError:
                if previous.exists():
                    previous.rename(destination)
                raise
    print(f"{len(files)} staged files; {len(changed)} changed, {len(stale)} stale")
    print(f"Output: {destination}")
    return 1 if not args.apply and (changed or stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())
