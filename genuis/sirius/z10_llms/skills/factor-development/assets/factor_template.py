"""因子定义: FACTOR_DEFINITION"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 使用 trade_time、code、REQUIRED_COLUMNS 计算 FACTOR_NAME 核心公式。"""
    return (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            FACTOR_EXPRESSION.alias('FACTOR_NAME')
        )
        .select(['trade_time', 'code', 'FACTOR_NAME'])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 FACTOR_NAME；输入必须包含 trade_time、code、REQUIRED_COLUMNS。"""
    return calculate(df_lazy)
