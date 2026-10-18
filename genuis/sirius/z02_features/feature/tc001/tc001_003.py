"""
因子代号: tc001_003
原历史名: tc001_003
因子定义: 动态买卖气指标 ADTM(23, 8) = (DTM均值 - DBM均值) / (DTM均值 + DBM均值)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算动态买卖气指标ADTM因子

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low、open
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc001_003 列
    """

    # 计算步骤:
    # 1. DTM = 当开盘价<昨收时取0，否则取max(最高价-开盘价, 开盘价-昨收)
    # 2. DBM = 当开盘价>=昨收时取0，否则取max(开盘价-最低价, 开盘价-昨收)
    # 3. STM = DTM的23周期累加，SBM = DBM的23周期累加
    # 4. ADTM = (STM-SBM)/STM 或 (STM-SBM)/SBM
    # 5. 使用 .over('code') 按品种分组
    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(
            # 前一周期开盘价
            pl.col('open').shift(1).over('code').alias('pre_open')
        )
        .with_columns(
            # DTM: 当开盘价<昨收时取0，否则取max(最高价-开盘价, 开盘价-昨收)
            pl.when(pl.col('open') < pl.col('pre_open'))
            .then(0.0)
            .otherwise(pl.max_horizontal([
                pl.col('high') - pl.col('open'),
                pl.col('open') - pl.col('pre_open')
            ]))
            .alias('dtm'),
            # DBM: 当开盘价>=昨收时取0，否则取max(开盘价-最低价, 开盘价-昨收)
            pl.when(pl.col('open') >= pl.col('pre_open'))
            .then(0.0)
            .otherwise(pl.max_horizontal([
                pl.col('open') - pl.col('low'),
                pl.col('open') - pl.col('pre_open')
            ]))
            .alias('dbm')
        )
        .with_columns(
            # STM = DTM的23周期累加
            pl.col('dtm').rolling_sum(23).over('code').alias('stm'),
            # SBM = DBM的23周期累加
            pl.col('dbm').rolling_sum(23).over('code').alias('sbm')
        )
        .with_columns(
            # ADTM = (STM-SBM)/STM 或 (STM-SBM)/SBM
            # 当STM>SBM时取(STM-SBM)/STM，STM=SBM时取0，否则取(STM-SBM)/SBM
            pl.when(pl.col('stm') > pl.col('sbm'))
            .then((pl.col('stm') - pl.col('sbm')) / pl.col('stm'))
            .when(pl.col('stm') == pl.col('sbm'))
            .then(0.0)
            .otherwise((pl.col('stm') - pl.col('sbm')) / pl.col('sbm'))
            .alias('adtm_raw')
        )
        .with_columns(
            # ADTM的8周期EMA作为平滑值
            pl.col('adtm_raw').ewm_mean(span=8).over('code').alias("tc001_003")
        )
        .select(['trade_time', 'code', "tc001_003"])
    )

    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """使用默认配置构造因子计算图。"""
    return calculate(df_lazy)
