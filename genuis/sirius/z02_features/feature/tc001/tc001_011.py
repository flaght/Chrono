"""
因子代号: tc001_011
原历史名: tc001_011
因子定义: 情绪指标 BRAR = sum(high - open, 26) / sum(open - low, 26) * 100
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算BRAR指标因子（输出AR值）

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、open
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_011 列（AR值）
    """

    # 计算步骤:
    # 1. AR = sum(High-Open, 26) / sum(Open-Low, 26) * 100
    # 2. BR = sum(High-YC, 26) / sum(YC-Low, 26) * 100
    # 3. 输出AR值作为因子
    # 4. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 上涨压力 = 最高价 - 开盘价
            (pl.col('high') - pl.col('open')).alias('up_pressure'),
            # 下跌压力 = 开盘价 - 最低价
            (pl.col('open') - pl.col('low')).alias('down_pressure')
        )
        .with_columns(
            # 26周期上涨压力累加
            pl.col('up_pressure').rolling_sum(26).over('code').alias('sum_up'),
            # 26周期下跌压力累加
            pl.col('down_pressure').rolling_sum(26).over('code').alias('sum_down')
        )
        .with_columns(
            # AR = sum(High-Open, 26) / sum(Open-Low, 26) * 100
            # 当分母=0时，AR设为100（中性值）
            pl.when(pl.col('sum_down') == 0)
            .then(100.0)
            .otherwise(pl.col('sum_up') / pl.col('sum_down') * 100)
            .alias("tc001_011")
        )
        .select(['trade_time', 'code', "tc001_011"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
