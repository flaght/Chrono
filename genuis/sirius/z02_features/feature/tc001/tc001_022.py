"""
因子代号: tc001_022
原历史名: tc001_022
因子定义: 下影线占比 = (min(open, close) - low) / (high - low)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算下影线比例因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、high、low、open
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_022 列
    """

    # 计算步骤:
    # 1. 下影线 = min(开盘价, 收盘价) - 最低价
    # 2. 下影线比例 = 下影线 / (最高价 - 最低价)
    # 3. 取值范围[0, 1]，越大表示下方支撑越强
    factor_lazy = (
        df_lazy
        .with_columns(
            # 下影线比例 = (min(开盘价, 收盘价) - 最低价) / (最高价 - 最低价)
            # 当最高价=最低价时，设为0
            pl.when(pl.col('high') == pl.col('low'))
            .then(0.0)
            .otherwise((pl.min_horizontal([pl.col('open'), pl.col('close')]) - 
                       pl.col('low')) / 
                      (pl.col('high') - pl.col('low')))
            .alias("tc001_022")
        )
        .select(['trade_time', 'code', "tc001_022"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
