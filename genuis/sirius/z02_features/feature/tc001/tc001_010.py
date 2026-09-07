"""
因子代号: tc001_010
原历史名: tc001_010
因子定义: 均势指标 BOP = (close - open) / (high - low)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算力量指标BOP因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、close、high、low、open
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_010 列
    """

    # 计算步骤:
    # 1. BOP = (收盘价 - 开盘价) / (最高价 - 最低价)
    # 2. 当最高价=最低价时，BOP设为0
    # 3. 取值范围[-1, 1]，正值表示买方力量强，负值表示卖方力量强
    factor_lazy = (
        df_lazy
        .with_columns(
            # BOP = (收盘价 - 开盘价) / (最高价 - 最低价)
            pl.when(pl.col('high') == pl.col('low'))
            .then(0.0)
            .otherwise((pl.col('close') - pl.col('open')) / 
                      (pl.col('high') - pl.col('low')))
            .alias("tc001_010")
        )
        .select(['trade_time', 'code', "tc001_010"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
