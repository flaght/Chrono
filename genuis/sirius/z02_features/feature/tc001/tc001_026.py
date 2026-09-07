"""
因子代号: tc001_026
原历史名: tc001_026
因子定义: 20周期价格动量 = close.shift(1) / close.shift(21) - 1
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算20周期动量因子（剔除最近1周期）

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_026 列
    """

    # 计算步骤:
    # 1. 计算20周期累计收益（从T-21到T-1），剔除最近1周期避免短期反转
    # 2. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 20周期动量 = (前一周期收盘价 / 21周期前收盘价) - 1
            # shift(1) 获取前一周期收盘价，shift(21) 获取21周期前收盘价
            (pl.col('close').shift(1).over('code') /
             pl.col('close').shift(21).over('code') - 1)
            .alias("tc001_026")
        )
        .select(['trade_time', 'code', "tc001_026"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
