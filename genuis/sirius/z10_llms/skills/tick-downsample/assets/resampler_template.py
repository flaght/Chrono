"""
算子名称: RESAMPLER_NAME
目标周期: 1min
因子分类: FEATURE_CATEGORY (如 future_1min, money_flow, order_flow, open_interest 等)
输入数据: CTP Level-1 Tick 快照流
输出规范: 标准 1 分钟微观宽表 (trade_time, code, ...)
规范对标: docs/tick_downsample_1min_features.md
"""

from __future__ import annotations
import polars as pl


def preprocess_ticks(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    阶段 1 & 2: 全局时序排序与 Tick 级衍生变量（增量、方向打标、订单流）
    """
    return (
        df_lazy
        .sort(["InstrumentID", "timestamp"])
        .with_columns([
            # 1. 基础差分增量（严格按标的分组隔离）
            pl.col("Volume").diff().clip(lower_bound=0).fill_null(0).over("InstrumentID").alias("_delta_v"),
            pl.col("Turnover").diff().clip(lower_bound=0).fill_null(0.0).over("InstrumentID").alias("_delta_m"),
            pl.col("OpenInterest").diff().fill_null(0).over("InstrumentID").alias("_delta_oi"),
            
            # 2. 盘口中间价与价差
            ((pl.col("AskPrice1") + pl.col("BidPrice1")) / 2.0).alias("_mid_price"),
            (pl.col("AskPrice1") - pl.col("BidPrice1")).alias("_spread"),
            
            # 3. 前一跳价格与报价（用于 Lee-Ready 与 OFI）
            pl.col("LastPrice").shift(1).over("InstrumentID").alias("_prev_last"),
            pl.col("AskPrice1").shift(1).over("InstrumentID").alias("_prev_ask"),
            pl.col("BidPrice1").shift(1).over("InstrumentID").alias("_prev_bid"),
            pl.col("AskVolume1").shift(1).over("InstrumentID").alias("_prev_ask_vol"),
            pl.col("BidVolume1").shift(1).over("InstrumentID").alias("_prev_bid_vol"),
        ])
        .with_columns([
            # 4. 改进型 Lee-Ready 主动买卖方向判定 D_t in {+1, -1, 0}
            pl.when(pl.col("_delta_v") == 0).then(pl.lit(0))
            .when(pl.col("LastPrice") >= pl.col("_prev_ask")).then(pl.lit(1))
            .when(pl.col("LastPrice") <= pl.col("_prev_bid")).then(pl.lit(-1))
            .when(pl.col("LastPrice") > pl.col("_prev_last")).then(pl.lit(1))
            .when(pl.col("LastPrice") < pl.col("_prev_last")).then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias("_trade_dir"),
        ])
    )


def aggregate_1min(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    阶段 3 & 4: 1分钟切片聚合与比率归一化
    """
    return (
        df_lazy
        # 生成 1 分钟切片时间标识
        .with_columns(
            pl.col("timestamp").dt.truncate("1m").alias("trade_time"),
            pl.col("InstrumentID").alias("code"),
        )
        .group_by(["trade_time", "code"])
        .agg([
            # 示例：基础 OHLCV
            pl.col("LastPrice").first().alias("open"),
            pl.col("LastPrice").max().alias("high"),
            pl.col("LastPrice").min().alias("low"),
            pl.col("LastPrice").last().alias("close"),
            pl.col("_delta_v").sum().alias("volume"),
            pl.col("_delta_m").sum().alias("money"),
            pl.len().alias("tick_count"),
            
            # 示例：主动买量与比率
            pl.col("_delta_v").filter(pl.col("_trade_dir") == 1).sum().alias("volume_in"),
            pl.col("_delta_v").filter(pl.col("_trade_dir") == -1).sum().alias("volume_out"),
        ])
        # 阶段 4: 比率归一化 (防除零)
        .with_columns([
            (pl.col("volume") / (pl.col("tick_count") + 1e-7)).alias("volume_per_tick"),
            (pl.col("volume_in") / (pl.col("volume") + 1e-7)).alias("volume_in_pct"),
            (pl.col("volume_out") / (pl.col("volume") + 1e-7)).alias("volume_out_pct"),
        ])
        .sort(["trade_time", "code"])
    )


def compute(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    外部入口：接收 Tick 懒加载流，构造纯 Polars 1分钟特征计算图
    """
    preprocessed = preprocess_ticks(df_lazy)
    return aggregate_1min(preprocessed)
