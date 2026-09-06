"""
因子代号: tc006_035
原历史名: tc006_035
因子定义: 单期对数收益率 LOG_RETURN(close, 1)
"""
import polars as pl

from feature.utils import log_return


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算单期对数收益率 LOG_RETURN_1。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_035 列
    """
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            log_return('close').alias("tc006_035")
        )
        .select(['trade_time', 'code', "tc006_035"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_035；输入含 trade_time、code、close。"""
    return calculate(df_lazy)
