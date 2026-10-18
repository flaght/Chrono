"""
因子代号: tc001_033
原历史名: tc001_033
因子定义: 5周期收益反转 = -(close / close.shift(5) - 1)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算5周期反转因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_033 列
    """

    # 计算步骤:
    # 1. 计算5周期累计收益
    # 2. 取负值作为反转因子（过去涨得多，未来可能跌）
    # 3. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 5周期反转 = -(5周期累计收益率)
            # 等价于: 1 - (当前周期收盘价 / 5周期前收盘价)
            (1 - pl.col('close') / pl.col('close').shift(5).over('code'))
            .alias("tc001_033")
        )
        .select(['trade_time', 'code', "tc001_033"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
