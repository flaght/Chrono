#!/usr/bin/env python3
"""静态检查 Orion 纯计算因子是否符合约定。"""

import ast
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
BATCH_RE = re.compile(r"^(?P<classification>[tm][cfsb])\d{3}$")
META_RE = re.compile(
    r"^因子定义:\s*(?P<definition>.+)$",
    re.DOTALL,
)
FORBIDDEN_CALLS = {
    "collect", "collect_async", "collect_schema", "fetch",
    "sink_parquet", "sink_csv", "sink_ipc", "sink_ndjson",
}
UNSUPPORTED_POLARS_METHODS = {
    "clip_min": "使用 `.clip(lower_bound=...)` 或 `pl.when(...).then(...).otherwise(...)`",
    "clip_max": "使用 `.clip(upper_bound=...)` 或 `pl.when(...).then(...).otherwise(...)`",
}
WINDOW_METHODS = {"shift", "diff", "pct_change"}
WINDOW_PREFIXES = ("rolling_", "ewm_", "cum_")
BASE_COLUMNS = {"trade_time", "code", "high", "low", "open", "close", "volume", "value", "openint"}
BASIS_COLUMNS = {
    "trade_time", "code", "future_open", "future_high", "future_low",
    "future_close", "future_volume", "future_value", "future_openint",
    "spot_open", "spot_high", "spot_low", "spot_close", "spot_volume", "spot_value",
}
LEGACY_DERIVED_COLUMNS = {"preClosePrice", "chgPct"}
UNSUPPORTED_COLUMNS = {"turnoverRate"}


def attribute_name(node):
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def annotation_text(node):
    return ast.unparse(node) if node is not None else ""


def module_string_constants(tree):
    values = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            values[node.targets[0].id] = node.value.value
    return values


def resolved_string(node, constants):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def is_window_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and (
            node.func.attr in WINDOW_METHODS
            or node.func.attr.startswith(WINDOW_PREFIXES)
        )
    )


def contains_nested_window(node):
    return is_window_call(node) and any(
        is_window_call(child)
        for child in ast.walk(node)
        if child is not node
    )


def validate(path):
    if path.name == "__init__.py":
        return []
    errors = []
    if path.suffix != ".py" or not NAME_RE.fullmatch(path.stem):
        errors.append("因子文件必须使用 lowercase snake_case 语义名称")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        return [f"无法解析文件: {exc}"]

    doc = (ast.get_docstring(tree, clean=False) or "").strip()
    match = META_RE.fullmatch(doc)
    factor_name = path.stem
    if not match:
        errors.append("模块文档必须以 `因子定义: ...` 开头")
    batch = path.parent.name
    batch_match = BATCH_RE.fullmatch(batch)
    if not batch_match:
        errors.append("所属批次必须使用 <家族><市场范围><三位批次号>")

    has_polars = any(
        isinstance(node, ast.Import)
        and any(alias.name == "polars" and alias.asname == "pl" for alias in node.names)
        for node in tree.body
    )
    if not has_polars:
        errors.append("缺少 `import polars as pl`")
    imports_loader = any(
        (isinstance(node, ast.ImportFrom) and node.module == "dataloader")
        or (isinstance(node, ast.Import)
            and any(alias.name == "dataloader" for alias in node.names))
        for node in tree.body
    )
    if imports_loader:
        errors.append("因子模块不得导入 dataloader；数据必须由外部注入")
    imports_formula_tree = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "feature.utils.formula"
        for node in tree.body
    )
    if imports_formula_tree:
        errors.append(
            "因子模块必须直接使用原生 Polars，不得导入 feature.utils.formula"
        )
    if any(isinstance(node, ast.ClassDef) for node in tree.body):
        errors.append("因子文件不得定义类")

    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    compute = next((node for node in functions if node.name == "compute"), None)
    calculate = next((node for node in functions if node.name == "calculate"), None)
    public = [node.name for node in functions if not node.name.startswith("_")]
    if compute is None:
        return errors + ["缺少 `compute(df_lazy)`"]
    if public not in (["compute"], ["calculate", "compute"]):
        errors.append(f"公开函数只能有 calculate 和 compute，当前为 {public}")

    args = compute.args
    parameterized = len(args.args) > 1
    if not args.args or args.args[0].arg != "df_lazy" or args.vararg or args.kwarg or args.kwonlyargs:
        errors.append("compute 第一个参数必须是 df_lazy，且不得使用可变参数")
    if len(args.defaults) != max(0, len(args.args) - 1):
        errors.append("compute 的所有附加参数必须有默认值")
    for default in args.defaults:
        if not isinstance(default, ast.Name) or not re.fullmatch(r"DEFAULT_[A-Z0-9_]+", default.id):
            errors.append("compute 的附加参数默认值必须引用 DEFAULT_* 常量")
    if annotation_text(args.args[0].annotation) != "pl.LazyFrame":
        errors.append("df_lazy 类型注解必须为 `pl.LazyFrame`")
    if annotation_text(compute.returns) != "pl.LazyFrame":
        errors.append("compute 返回类型必须标注为 `pl.LazyFrame`")

    if parameterized and calculate is None:
        errors.append("参数化因子必须使用 calculate 实现因子逻辑")
    if calculate is not None:
        calculate_args = calculate.args
        if calculate_args.vararg or calculate_args.kwarg or calculate_args.kwonlyargs:
            errors.append("calculate 不得定义可变参数或仅关键字参数")
        if annotation_text(calculate.returns) not in {"pl.Expr", "pl.LazyFrame"}:
            errors.append("calculate 返回类型必须标注为 `pl.Expr` 或 `pl.LazyFrame`")

    logic = calculate or compute
    all_functions = [node for node in (calculate, compute) if node is not None]
    all_calls = [node for function in all_functions for node in ast.walk(function) if isinstance(node, ast.Call)]
    logic_calls = [node for node in ast.walk(logic) if isinstance(node, ast.Call)]
    compute_calls = [node for node in ast.walk(compute) if isinstance(node, ast.Call)]
    forbidden = sorted({attribute_name(call.func) for call in all_calls} & FORBIDDEN_CALLS)
    if forbidden:
        errors.append(f"因子内部不得触发执行或写出，发现调用: {forbidden}")
    unsupported_polars = sorted(
        {attribute_name(call.func) for call in all_calls} & set(UNSUPPORTED_POLARS_METHODS)
    )
    for method in unsupported_polars:
        errors.append(
            f"当前 Polars 运行契约不支持 `.{method}()`；"
            f"{UNSUPPORTED_POLARS_METHODS[method]}"
        )
    forbidden_schema_attributes = sorted({
        node.attr
        for function in all_functions
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "df_lazy"
        and node.attr in {"columns", "schema"}
    })
    if forbidden_schema_attributes:
        errors.append(
            "因子内部不得检查输入 Schema；由外部调度器负责，发现: "
            f"df_lazy.{', df_lazy.'.join(forbidden_schema_attributes)}"
        )
    if any(isinstance(call.func, ast.Name) and call.func.id == "dataloader" for call in all_calls):
        errors.append("因子内部不得调用 dataloader")
    if any(
        isinstance(call.func, ast.Name) and call.func.id == "compile_formula"
        for call in all_calls
    ):
        errors.append("因子模块不得调用 compile_formula")
    nested_window_lines = sorted({
        call.lineno for call in all_calls if contains_nested_window(call)
    })
    if nested_window_lines:
        errors.append(
            "时序表达式不得直接嵌套；请使用多个 with_columns 阶段拆分，"
            f"涉及行: {nested_window_lines}"
        )
    if any(
        isinstance(node, ast.Name) and node.id in {"begDate", "endDate"}
        for function in all_functions for node in ast.walk(function)
    ):
        errors.append("因子计算不得依赖 begDate 或 endDate")

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
    registered_columns = BASE_COLUMNS | BASIS_COLUMNS
    legacy_columns = sorted(referenced_columns & LEGACY_DERIVED_COLUMNS)
    if legacy_columns:
        errors.append(f"旧字段 {legacy_columns} 不得作为输入列引用；必须从 close 派生")
    unsupported_columns = sorted(referenced_columns & UNSUPPORTED_COLUMNS)
    if unsupported_columns:
        errors.append(f"当前不支持依赖字段: {unsupported_columns}")
    unknown_columns = sorted(
        referenced_columns - registered_columns - LEGACY_DERIVED_COLUMNS
        - UNSUPPORTED_COLUMNS - aliases - {factor_name}
    )
    if unknown_columns:
        errors.append(f"引用了未登记的基础字段: {unknown_columns}")

    doc = ast.get_docstring(compute if parameterized else logic) or ""
    required_doc_fields = {"df_lazy", "pl.LazyFrame", "trade_time", "code", factor_name}
    required_doc_fields.update(referenced_columns & registered_columns)
    for text in sorted(required_doc_fields):
        if text not in doc:
            errors.append(f"compute 文档缺少 {text!r}")

    constants = module_string_constants(tree)
    expected = ["trade_time", "code", factor_name]
    output_calls = logic_calls + compute_calls
    exact_select = any(
        attribute_name(call.func) == "select"
        and len(call.args) == 1
        and isinstance(call.args[0], (ast.List, ast.Tuple))
        and [resolved_string(item, constants) for item in call.args[0].elts] == expected
        for call in output_calls
    )
    has_parameterized_select = any(
        attribute_name(call.func) == "select"
        and len(call.args) == 1
        and isinstance(call.args[0], (ast.List, ast.Tuple))
        and [resolved_string(item, constants) for item in call.args[0].elts[:2]]
        == ["trade_time", "code"]
        for call in output_calls
    )
    if not exact_select and not (parameterized and has_parameterized_select):
        errors.append(
            "calculate 或 compute 必须最终选择 trade_time、code 和因子列"
        )
    has_factor_alias = any(
        attribute_name(call.func) == "alias"
        and call.args
        and resolved_string(call.args[0], constants) == factor_name
        for call in output_calls
    )
    if not has_factor_alias and not parameterized:
        errors.append(f"calculate 或 compute 缺少最终 `.alias({factor_name!r})`")

    if parameterized:
        if not any(
            isinstance(call.func, ast.Name) and call.func.id == "calculate"
            for call in compute_calls
        ):
            errors.append("参数化 compute 必须调用 calculate 生成单参数因子逻辑")
    return errors


def main():
    if len(sys.argv) < 2:
        print("用法: validate_factor.py 因子文件.py [因子文件.py ...]", file=sys.stderr)
        return 2
    failed = False
    for raw in sys.argv[1:]:
        path = Path(raw)
        errors = validate(path)
        print(("失败" if errors else "通过"), path)
        for error in errors:
            print(f"  - {error}")
        failed |= bool(errors)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
