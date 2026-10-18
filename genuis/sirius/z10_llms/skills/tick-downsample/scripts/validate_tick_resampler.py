#!/usr/bin/env python3
"""
静态检查 Orion Tick 降频与微观特征算子是否符合 Polars Lazy 规范。
"""

import ast
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
FORBIDDEN_CALLS = {
    "collect", "collect_async", "fetch", "sink_parquet", "sink_csv",
    "sink_ipc", "sink_ndjson", "to_pandas", "to_numpy", "iter_rows",
}

REQUIRED_TICK_COLUMNS = {
    "TradingDay", "InstrumentID", "UpdateTime", "UpdateMillisec",
    "LastPrice", "Volume", "Turnover", "AveragePrice",
    "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1",
    "OpenInterest", "UpperLimitPrice", "LowerLimitPrice",
    "timestamp",
}


def attribute_name(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def validate(path: Path) -> list[str]:
    if path.name == "__init__.py":
        return []
    errors = []
    if path.suffix != ".py" or not NAME_RE.fullmatch(path.stem):
        errors.append("降频算子文件必须使用 lowercase snake_case 语义名称")

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        return [f"无法解析文件: {exc}"]

    # 1. 检查 import polars as pl
    has_polars = any(
        isinstance(node, ast.Import)
        and any(alias.name == "polars" and alias.asname == "pl" for alias in node.names)
        for node in tree.body
    )
    if not has_polars:
        errors.append("缺少 `import polars as pl`")

    # 2. 检查是否有禁用类定义
    if any(isinstance(node, ast.ClassDef) for node in tree.body):
        errors.append("降频算子模块应遵循纯函数式设计，不得定义类")

    # 3. 检查函数入口
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    compute = next((node for node in functions if node.name == "compute"), None)
    if compute is None:
        errors.append("缺少主入口函数 `compute(df_lazy, ...)`")
    else:
        args = compute.args
        if not args.args or args.args[0].arg != "df_lazy":
            errors.append("compute 第一个参数必须为 `df_lazy`")

    # 4. 检查禁止的 Eager 调用
    all_calls = [node for function in functions for node in ast.walk(function) if isinstance(node, ast.Call)]
    forbidden = sorted({attribute_name(call.func) for call in all_calls} & FORBIDDEN_CALLS)
    if forbidden:
        errors.append(f"降频算子内部不得触发 Eager 执行或写出，发现调用: {forbidden}")

    # 5. 检查引用的列名是否在合法范围或由临时别名产生
    aliases = {
        call.args[0].value for call in all_calls
        if attribute_name(call.func) == "alias" and call.args
        and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)
    }
    referenced_columns = {
        call.args[0].value for call in all_calls
        if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "pl" and call.func.attr == "col"
        and call.args and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }

    # 允许的列包括：输入字段、临时别名、中间特征名
    unknown_columns = sorted(referenced_columns - REQUIRED_TICK_COLUMNS - aliases)
    # 过滤掉非标准但由 group_by 产生或特殊引用的部分
    common_bar_keys = {"trade_time", "code", "bar_time"}
    unknown_columns = [col for col in unknown_columns if col not in common_bar_keys]
    if unknown_columns:
        errors.append(f"引用了未在 CTP 契约中登记的输入字段: {unknown_columns}")

    return errors


def main():
    if len(sys.argv) < 2:
        print("用法: python3 validate_tick_resampler.py <path_to_resampler.py>")
        sys.exit(1)

    path = Path(sys.argv[1])
    errors = validate(path)
    if errors:
        print(f"❌ 校验未通过 [{path}]:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"✅ 校验通过 [{path}]: 严格符合 Polars Lazy 降频算子规范")
        sys.exit(0)


if __name__ == "__main__":
    main()
