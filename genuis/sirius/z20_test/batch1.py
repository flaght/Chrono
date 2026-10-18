"""通过 t001 包的显式导出批量计算因子。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl

# FACTORS 由 feature/t001/__init__.py 中的 *_compute 显式注册生成。
import feature.tc001 as i001

KEY_COLUMNS = ["trade_time", "code"]


def compute_batch(
    df_lazy: pl.LazyFrame,
    names: Iterable[str] | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
) -> pl.LazyFrame:
    """
    使用 t001 包导出的 compute 函数批量构造因子计算图。

    参数:
        df_lazy: 外部已经加载的基础数据
        names: 需要计算的因子名；为 None 时计算 t001 全部因子
        params: 按因子名传入 compute 参数
    返回:
        包含 trade_time、code 和全部选定因子列的 pl.LazyFrame
    """
    FACTORS = i001.__all__
    selected_names = list(FACTORS) if names is None else list(names)
    if not selected_names:
        raise ValueError("至少需要指定一个因子")

    duplicate_names = sorted(
        name for name, count in Counter(selected_names).items() if count > 1
    )
    if duplicate_names:
        raise ValueError(f"因子名不能重复: {duplicate_names}")

    unknown_names = sorted(set(selected_names) - FACTORS.keys())
    if unknown_names:
        raise KeyError(f"不存在的因子: {unknown_names}")

    factor_params = {} if params is None else dict(params)
    unknown_param_names = sorted(set(factor_params) - set(selected_names))
    if unknown_param_names:
        raise KeyError(f"参数指定了未选中的因子: {unknown_param_names}")

    factor_frames = [
        FACTORS[name](df_lazy, **dict(factor_params.get(name, {})))
        for name in selected_names
    ]

    if len(factor_frames) == 1:
        return factor_frames[0].sort(KEY_COLUMNS)

    # 每个 compute 均返回公共键和自身因子列，align 按公共键合并为宽表。
    return pl.concat(factor_frames, how="align").sort(KEY_COLUMNS)


def compute_file(
    file_path: str | Path,
    names: Iterable[str] | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
) -> pl.LazyFrame:
    """懒加载 Feather/Arrow IPC 文件并批量构造 t001 因子计算图。"""
    df_lazy = pl.scan_ipc(file_path)
    return compute_batch(df_lazy, names=names, params=params)


def collect_file(
    file_path: str | Path,
    names: Iterable[str] | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    """批量计算指定文件中的 t001 因子并统一执行一次 collect。"""
    return compute_file(file_path, names=names, params=params).collect()


def diagnose_file(
    file_path: str | Path,
    names: Iterable[str] | None = None,
    params: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, str]:
    """
    逐个执行因子并返回失败信息，仅用于定位批量计算错误。

    返回空字典表示全部因子均可独立执行；诊断会分别 collect，不能替代正式批量计算。
    """
    selected_names = list(FACTORS) if names is None else list(names)
    factor_params = {} if params is None else dict(params)
    unknown_names = sorted(set(selected_names) - FACTORS.keys())
    if unknown_names:
        raise KeyError(f"不存在的因子: {unknown_names}")

    df_lazy = pl.scan_ipc(file_path)
    failures: dict[str, str] = {}

    for name in selected_names:
        try:
            FACTORS[name](
                df_lazy,
                **dict(factor_params.get(name, {})),
            ).collect()
        except Exception as exc:  # 诊断入口需要保留 Polars 的原始异常信息
            failures[name] = f"{type(exc).__name__}: {exc}"

    return failures



file_path = "/workspace/worker/pj/Chrono/genuis/mizar/records/ricso2/rbb/basic/train_data.feather"
data = collect_file(file_path)
print(data.tail())
