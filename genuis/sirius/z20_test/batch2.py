"""按 feature 顶层注册表分批计算因子的调用示例。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import polars as pl

from feature import BATCH_FACTORS
from feature.utils.common import compute_factor_batch


DEFAULT_BATCHES = (
    "tc001",
    "tc002",
    "tc003",
    "tc004",
    "tc005",
    "tf001",
    "tf002",
)

FactorCallback = Callable[..., None]
FactorParams = Mapping[str, Mapping[str, object]]
BatchParams = Mapping[str, FactorParams]


def run_batch(
    data: pl.LazyFrame,
    batch: str,
    callback: FactorCallback,
    instruments: Iterable[str],
    method: str,
    start_date: str,
    end_date: str,
    *,
    names: Iterable[str] | None = None,
    params: FactorParams | None = None,
) -> None:
    """计算一个批次，并通过 callback 返回该批次的 Polars DataFrame。"""
    if batch not in BATCH_FACTORS:
        raise KeyError(f"不存在的因子批次: {batch}")

    factors = BATCH_FACTORS[batch]
    selected = tuple(factors) if names is None else tuple(names)
    print(f"计算批次 {batch}: {len(selected)} 个因子")

    factors_data = compute_factor_batch(
        data,
        factors,
        names=selected,
        params=params,
    ).collect()

    callback(
        factors_data=factors_data,
        instruments=tuple(instruments),
        name=batch,
        method=method,
        start_date=start_date,
        end_date=end_date,
    )


def calculate_factors(
    data: pl.LazyFrame,
    callback: FactorCallback,
    instruments: Iterable[str],
    method: str,
    start_date: str,
    end_date: str,
    *,
    batches: Iterable[str] = DEFAULT_BATCHES,
    names_by_batch: Mapping[str, Iterable[str]] | None = None,
    params_by_batch: BatchParams | None = None,
) -> None:
    """按照批次顺序计算因子；每个批次只执行一次 collect。"""
    selected_names = {} if names_by_batch is None else names_by_batch
    selected_params = {} if params_by_batch is None else params_by_batch

    for batch in batches:
        run_batch(
            data=data,
            batch=batch,
            callback=callback,
            instruments=instruments,
            method=method,
            start_date=start_date,
            end_date=end_date,
            names=selected_names.get(batch),
            params=selected_params.get(batch),
        )


def print_callback(**result: Any) -> None:
    """示例回调；实际使用时可以替换为入库或写文件函数。"""
    factors_data: pl.DataFrame = result["factors_data"]
    print(
        f"完成 {result['name']}: "
        f"rows={factors_data.height}, columns={factors_data.width}"
    )


if __name__ == "__main__":
    file_path = (
        "/workspace/worker/pj/Chrono/genuis/mizar/records/"
        "ricso2/rbb/basic/recent_data.feather"
    )
    df_lazy = pl.scan_ipc(file_path)

    calculate_factors(
        data=df_lazy,
        callback=print_callback,
        instruments=("RB",),
        method="factor_batch",
        start_date="2012-01-01",
        end_date="2025-04-30",
        # 只测试部分批次时可改为：batches=("tc005", "tf001", "tf002")
        batches=DEFAULT_BATCHES,
        # 按批次选择部分因子；未指定的批次默认计算该批次全部因子。
        names_by_batch={
            "tf001": ("oi001", "oi006"),
            "tf002": ("cr046", "cr053"),
        },
        # 参数按“批次 -> 因子名 -> compute 参数”组织。
        params_by_batch={
            "tf001": {
                "oi001": {"period": 15},
                "oi006": {"period": 15},
            },
            "tf002": {
                "cr046": {"period": 15},
                "cr053": {"periods": (5, 15)},
            },
        },
    )
