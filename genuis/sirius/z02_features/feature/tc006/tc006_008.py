"""
因子代号: tc006_008
原历史名: tc006_008
因子定义: 阿隆震荡指标 AROON_UP(25) - AROON_DOWN(25)
"""
import polars as pl


def calculate(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    计算 25 周期阿隆震荡指标 AROON_OSC。

    参数:
        df_lazy: pl.LazyFrame，必须包含 trade_time、code、high、low
    返回:
        pl.LazyFrame: 包含 trade_time, code, tc006_008 列
    """
    # 计算过去 25 周期最高点和最低点出现的滞后位置 (0..24)
    # 使用 rolling_max 和 shift 匹配
    high_shifts = [pl.col('high').shift(i).over('code').alias(f'_h_{i}') for i in range(25)]
    low_shifts = [pl.col('low').shift(i).over('code').alias(f'_l_{i}') for i in range(25)]

    factor_lazy = (
        df_lazy
        .sort(by=['trade_time', 'code'])
        .with_columns(high_shifts + low_shifts)
        .with_columns(
            pl.col('high').rolling_max(25).over('code').alias('_max_h'),
            pl.col('low').rolling_min(25).over('code').alias('_min_l'),
        )
        .with_columns(
            # 找到最近最高点距离当前周期的滞后 K 线数
            pl.min_horizontal([
                pl.when(pl.col(f'_h_{i}') == pl.col('_max_h')).then(float(i)).otherwise(25.0)
                for i in range(25)
            ]).alias('_bars_high'),
            # 找到最近最低点距离当前周期的滞后 K 线数
            pl.min_horizontal([
                pl.when(pl.col(f'_l_{i}') == pl.col('_min_l')).then(float(i)).otherwise(25.0)
                for i in range(25)
            ]).alias('_bars_low'),
        )
        .with_columns(
            ((25.0 - pl.col('_bars_high')) / 25.0 * 100.0).alias('_aroon_up'),
            ((25.0 - pl.col('_bars_low')) / 25.0 * 100.0).alias('_aroon_down'),
        )
        .with_columns(
            (pl.col('_aroon_up') - pl.col('_aroon_down')).alias("tc006_008")
        )
        .select(['trade_time', 'code', "tc006_008"])
    )
    return factor_lazy


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """从 df_lazy: pl.LazyFrame 计算 tc006_008；输入含 trade_time、code、high、low。"""
    return calculate(df_lazy)
