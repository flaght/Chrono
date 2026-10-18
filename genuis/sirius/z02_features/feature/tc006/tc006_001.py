"""
因子代号: tc006_001
原历史名: tc006_001
因子定义: 双指数移动平均线比率偏离 EMA(close, fast) / EMA(close, slow) - 1
"""
from collections.abc import Iterable
import polars as pl

from feature.utils import safe_div

DEFAULT_PERIOD_PAIRS = ((7, 21), (12, 26))


def calculate(
    periods: tuple[int, int],
) -> pl.Expr:
    """
    实现单个快慢均线周期对的 EMA 偏离率表达式。

    参数:
        periods: (fast, slow) 正整数周期元组，要求 fast < slow
    返回:
        pl.Expr: EMA(close, fast) / EMA(close, slow) - 1
    """
    fast, slow = periods
    fast_ema = pl.col('close').ewm_mean(span=fast, adjust=False).over('code')
    slow_ema = pl.col('close').ewm_mean(span=slow, adjust=False).over('code')
    return safe_div(fast_ema, slow_ema) - 1.0


def compute(
    df_lazy: pl.LazyFrame,
    period_pairs: Iterable[tuple[int, int]] = DEFAULT_PERIOD_PAIRS,
) -> pl.LazyFrame:
    """
    对外构造 tc006_001 因子计算图，未传参数时使用 DEFAULT_PERIOD_PAIRS。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
        period_pairs: 快慢均线周期元组集合
    返回:
        pl.LazyFrame: 包含 trade_time、code 和 tb001_<fast>_<slow> 因子列
    """
    normalized_pairs = tuple(dict.fromkeys(period_pairs))
    if not normalized_pairs:
        raise ValueError("至少需要一个周期对")
    for pair in normalized_pairs:
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise ValueError("每个周期对必须为包含两个正整数的元组")
        fast, slow = pair
        if not (isinstance(fast, int) and isinstance(slow, int) and fast > 0 and slow > 0 and fast < slow):
            raise ValueError("快慢周期必须为正整数且 fast < slow")

    factor_names = [f"tc006_001_{fast}_{slow}" for fast, slow in normalized_pairs]
    expressions = [
        calculate(pair).alias(name)
        for pair, name in zip(normalized_pairs, factor_names)
    ]

    return (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(expressions)
        .select(['trade_time', 'code', *factor_names])
    )
