"""
因子代号: tc001_038
原历史名: tc001_038
因子定义: 威廉变异离散量 WVAD = rolling_sum(((close - open) / (high - low)) * volume, 24)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算威廉变异离散量WVAD因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、high、low、open、volume
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_038 列
    """

    # 计算步骤:
    # 1. WVAD = 成交量 * (收盘价 - 开盘价) / (最高价 - 最低价)
    # 2. 当最高价=最低价时，WVAD设为0
    # 3. 正值表示买方力量强，负值表示卖方力量强
    factor_lazy = (
        df_lazy
        .with_columns(
            # WVAD = 成交量 * (收盘价 - 开盘价) / (最高价 - 最低价)
            pl.when(pl.col('high') == pl.col('low'))
            .then(0.0)
            .otherwise(pl.col('volume') * 
                      (pl.col('close') - pl.col('open')) / 
                      (pl.col('high') - pl.col('low')))
            .alias("tc001_038")
        )
        .select(['trade_time', 'code', "tc001_038"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
