# 外部模型代码生成骨架

这些骨架提炼自 `/Users/kerry/work/orion/feature/` 中已经通过成熟 `factor-development` 校验器的代码风格。它们不是新的业务契约，只把校验器要求转成不易误解的代码形式。

## 参数化复杂因子（优先）

```python
"""因子定义: FACTOR_DEFINITION"""
import polars as pl

DEFAULT_PERIOD = 20
NAME = "FACTOR_NAME"


def calculate(df_lazy: pl.LazyFrame, period: int) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 使用 trade_time、code、close 计算 FACTOR_NAME 核心公式。"""
    return (
        df_lazy
        .with_columns(
            pl.col("close").shift(period).over("code").alias("_previous_close")
        )
        .with_columns(
            pl.when(pl.col("_previous_close") != 0)
            .then(pl.col("close") / pl.col("_previous_close") - 1.0)
            .otherwise(None)
            .alias(NAME)
        )
        .select(["trade_time", "code", NAME])
    )


def compute(
    df_lazy: pl.LazyFrame,
    period: int = DEFAULT_PERIOD,
) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 FACTOR_NAME；输入必须包含 trade_time、code、close。"""
    if not isinstance(period, int) or isinstance(period, bool) or period <= 0:
        raise ValueError("period 必须是正整数")
    return calculate(df_lazy.sort(["trade_time", "code"]), period)
```

多个参数时展开为独立参数和独立默认常量：

```python
DEFAULT_OI_LOOKBACK = 5
DEFAULT_MOM_LOOKBACK = 10
DEFAULT_STD_LOOKBACK = 120
DEFAULT_THRESHOLD = 1.0

def calculate(
    df_lazy: pl.LazyFrame,
    oi_lookback: int,
    mom_lookback: int,
    std_lookback: int,
    threshold: float,
) -> pl.LazyFrame:
    ...

def compute(
    df_lazy: pl.LazyFrame,
    oi_lookback: int = DEFAULT_OI_LOOKBACK,
    mom_lookback: int = DEFAULT_MOM_LOOKBACK,
    std_lookback: int = DEFAULT_STD_LOOKBACK,
    threshold: float = DEFAULT_THRESHOLD,
) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 FACTOR_NAME；输入必须包含 trade_time、code、close、openint。"""
    ...
```

## 无参数因子

```python
"""因子定义: FACTOR_DEFINITION"""
import polars as pl

def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 使用 trade_time、code、close 计算 FACTOR_NAME 核心公式。"""
    return (
        df_lazy
        .with_columns(FACTOR_EXPRESSION.alias("FACTOR_NAME"))
        .select(["trade_time", "code", "FACTOR_NAME"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 FACTOR_NAME；输入必须包含 trade_time、code、close。"""
    return calculate(df_lazy.sort(["trade_time", "code"]))
```

## 生成前自检

- 函数签名内存在 `df_lazy: pl.LazyFrame` 和 `-> pl.LazyFrame`；
- 参数化 `calculate` 返回 `pl.LazyFrame` 或 `pl.Expr`；
- `compute` docstring 包含精确因子名和全部必需字段；
- 所有附加默认值都是 `DEFAULT_*` 名称；
- 最终只选择 `trade_time`、`code` 和因子列；
- 不使用已移除的 `.clip_min()`、`.clip_max()`；分别改为 `.clip(lower_bound=...)`、`.clip(upper_bound=...)` 或显式 `pl.when`；
- 没有 `.collect()`、I/O、数据加载或网络调用。
