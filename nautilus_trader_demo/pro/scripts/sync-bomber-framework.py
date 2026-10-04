#!/usr/bin/env python3
"""Sync the pro framework into bomber.framework, rewriting Python imports.

Run from any directory. Default output is <pro>/dist/bomber/framework.
By default this reports drift; --apply rebuilds the staged copy.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import shutil
import tempfile
import re
import sys
from pathlib import Path

PACKAGE_PATHS = {"market": "market", "trader": "trader", "datahub": "datahub"}
IMPORT_ROOTS = {"market": "market", "trader": "trader", "datahub": "datahub", "fixes": "fixes"}
EXCLUDED_DIRS = {".build-deps", "__pycache__", "build", "dist", ".venv"}
DEFAULT_SOURCE = Path(__file__).resolve().parents[1]
STAGED_SUFFIXES = {".py", ".pyx", ".pxd"}
MARKER = ".sync-bomber-framework"


def old_imports(tree: ast.AST) -> list[str]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".", 1)[0] in IMPORT_ROOTS:
                found.append(f"line {node.lineno}: from {node.module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in IMPORT_ROOTS:
                    found.append(f"line {node.lineno}: import {alias.name}")
    return found


def convert(source: str, path: Path) -> str:
    """Rewrite absolute imports only; preserve all other source text."""
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines(keepends=True)
    edits: dict[int, list[tuple[int, int, str]]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".", 1)[0] in IMPORT_ROOTS:
                line = lines[node.lineno - 1]
                match = re.search(r"\bfrom\s+" + re.escape(node.module) + r"(?=\s+import\b)", line)
                if not match:
                    raise ValueError(f"cannot locate import at {path}:{node.lineno}")
                start = match.start() + match.group().index(node.module)
                edits.setdefault(node.lineno - 1, []).append((start, start + len(node.module), "bomber.framework." + IMPORT_ROOTS[node.module.split(".", 1)[0]] + node.module[len(node.module.split(".", 1)[0]):]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] not in IMPORT_ROOTS:
                    continue
                line = lines[node.lineno - 1]
                match = re.search(r"(?<![\w.])" + re.escape(alias.name) + r"(?=\s|,|$)", line)
                if not match:
                    raise ValueError(f"cannot locate import at {path}:{node.lineno}")
                edits.setdefault(node.lineno - 1, []).append((match.start(), match.end(), "bomber.framework." + IMPORT_ROOTS[alias.name.split(".", 1)[0]] + alias.name[len(alias.name.split(".", 1)[0]):]))
    for index, replacements in edits.items():
        for start, end, replacement in sorted(replacements, reverse=True):
            lines[index] = lines[index][:start] + replacement + lines[index][end:]
    result = "".join(lines)
    remaining = old_imports(ast.parse(result, filename=str(path)))
    if remaining:
        raise ValueError(f"unconverted imports in {path}: {remaining}")
    return result


def staged_files(source_root: Path) -> dict[Path, str]:
    files: dict[Path, str] = {
        Path("__init__.py"): (
            '"""Bomber framework staged from pro."""\n'
            '__all__ = ["datahub", "market", "trader"]\n'
        ),
    }
    for source_package, destination_package in PACKAGE_PATHS.items():
        source_dir = source_root / source_package
        if not source_dir.is_dir():
            raise ValueError(f"missing source package: {source_dir}")
        for source in sorted(source_dir.rglob("*")):
            if not source.is_file() or source.suffix not in STAGED_SUFFIXES:
                continue
            if source.name == "setup.py" or EXCLUDED_DIRS.intersection(source.relative_to(source_dir).parts):
                continue
            relative = Path(destination_package) / source.relative_to(source_dir)
            content = source.read_text(encoding="utf-8")
            files[relative] = convert(content, source) if source.suffix == ".py" else content
    fixes = source_root / "fixes.py"
    if not fixes.is_file():
        raise ValueError(f"missing source file: {fixes}")
    files[Path("fixes.py")] = convert(fixes.read_text(encoding="utf-8"), fixes)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dist-root", type=Path, help="output root; defaults to SOURCE/dist")
    parser.add_argument("--apply", action="store_true", help="build the staged copy; default is read-only")
    parser.add_argument("--diff", action="store_true", help="show unified diffs for changed files")
    args = parser.parse_args()
    source_root = args.source.resolve()
    if not source_root.is_dir():
        parser.error(f"source directory does not exist: {source_root}")
    dist_root = (args.dist_root or source_root / "dist").resolve()
    destination_root = dist_root / "bomber" / "framework"
    if dist_root == source_root or dist_root in (source_root / name for name in PACKAGE_PATHS):
        parser.error("dist root must be separate from source packages")
    try:
        files = staged_files(source_root)
    except (OSError, SyntaxError, ValueError) as exc:
        parser.error(str(exc))
    existing = set()
    if destination_root.exists():
        if not destination_root.is_dir() or destination_root.is_symlink():
            parser.error(f"staging destination is not a regular directory: {destination_root}")
        if not (destination_root / MARKER).is_file():
            parser.error(f"refusing to replace unmarked directory: {destination_root}")
        existing = {p.relative_to(destination_root) for p in destination_root.rglob("*") if p.is_file() and p.name != MARKER}
    changed = []
    for relative, converted in sorted(files.items()):
        destination = destination_root / relative
        current = destination.read_text(encoding="utf-8") if relative in existing else ""
        if converted == current:
            continue
        changed.append(relative)
        print(f"{'STAGE' if args.apply else 'DRIFT'} {relative}")
        if args.diff:
            sys.stdout.writelines(difflib.unified_diff(
                current.splitlines(keepends=True), converted.splitlines(keepends=True),
                fromfile=str(destination), tofile=str(source_root / relative),
            ))
    stale = sorted(existing - files.keys())
    for relative in stale:
        print(f"{'REMOVE' if args.apply else 'STALE'} {relative}")
    if args.apply:
        parent = destination_root.parent
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".framework-stage-", dir=parent) as temporary:
            staged = Path(temporary) / "framework"
            staged.mkdir()
            for relative, content in files.items():
                output = staged / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(content, encoding="utf-8")
            (staged / MARKER).write_text("Generated by sync-bomber-framework.py; do not edit.\n", encoding="utf-8")
            if destination_root.exists():
                shutil.rmtree(destination_root)
            staged.rename(destination_root)
    print(f"{len(files)} staged source files; {len(changed)} changed, {len(stale)} stale")
    print(f"Output: {destination_root}")
    return 1 if not args.apply and (changed or stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())
