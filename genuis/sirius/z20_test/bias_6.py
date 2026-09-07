import polars as pl

def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算6周期乖离率因子。

    参数:
        df_lazy: pl.LazyFrame，必须包含 date、Code、closePrice
    返回:
        pl.LazyFrame: 包含 date、Code、bias_6 列
    """
    # 计算步骤:
    # 1. 按股票计算6周期收盘价移动平均线
    # 2. 计算收盘价相对移动平均线的百分比偏离
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            pl.col('close').rolling_mean(6).over('code').alias('ma_6')
        )
        .with_columns(
            pl.when(pl.col('ma_6') == 0)
            .then(None)
            .otherwise(
                (pl.col('close') - pl.col('ma_6'))
                / pl.col('ma_6')
                * 100
            )
            .alias('bias_6')
        )
        .select(['trade_time', 'code', 'bias_6'])
    )

    return factor_lazy