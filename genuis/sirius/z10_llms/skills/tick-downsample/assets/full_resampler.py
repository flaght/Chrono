"""
全量 1 分钟微观特征降频引擎 (Full 1-Minute Microstructure Resampler)

基于 CTP Level-1 原始快照流，以纯 Polars Lazy 链式表达式，
一次性高并行度计算 8 大分类、93 个特征指标。

输入字段要求:
    TradingDay, InstrumentID, UpdateTime, UpdateMillisec, LastPrice,
    Volume, Turnover, AveragePrice, BidPrice1, BidVolume1, AskPrice1,
    AskVolume1, OpenInterest, UpperLimitPrice, LowerLimitPrice

规范对标: docs/tick_downsample_1min_features.md
"""

from __future__ import annotations
import math
import polars as pl


def build_tick_primitives(df_lazy: pl.LazyFrame, multiplier: float = 10.0) -> pl.LazyFrame:
    """
    阶段 1 & 2: 全局 Tick 衍生变量构造与向量化打标
    """
    eps = 1e-7

    return (
        df_lazy
        .sort(["InstrumentID", "timestamp"])
        .with_columns([
            # 1. 基础单跳差分 (按标的分组隔离)
            pl.col("Volume").diff().clip(lower_bound=0).fill_null(0).over("InstrumentID").alias("_delta_v"),
            pl.col("Turnover").diff().clip(lower_bound=0).fill_null(0.0).over("InstrumentID").alias("_delta_m"),
            pl.col("OpenInterest").diff().fill_null(0).over("InstrumentID").alias("_delta_oi"),
            
            # 2. 滞后一跳状态
            pl.col("LastPrice").shift(1).over("InstrumentID").alias("_prev_last"),
            pl.col("AskPrice1").shift(1).over("InstrumentID").alias("_prev_ask"),
            pl.col("BidPrice1").shift(1).over("InstrumentID").alias("_prev_bid"),
            pl.col("AskVolume1").shift(1).over("InstrumentID").alias("_prev_ask_vol"),
            pl.col("BidVolume1").shift(1).over("InstrumentID").alias("_prev_bid_vol"),
            
            # 3. 盘口中间价与价差
            ((pl.col("AskPrice1") + pl.col("BidPrice1")) / 2.0).alias("_mid_price"),
            (pl.col("AskPrice1") - pl.col("BidPrice1")).alias("_spread"),
        ])
        .with_columns([
            # 4. 微观价与对数收益率
            (
                (pl.col("AskPrice1") * pl.col("BidVolume1") + pl.col("BidPrice1") * pl.col("AskVolume1"))
                / (pl.col("BidVolume1") + pl.col("AskVolume1") + eps)
            ).alias("_micro_price"),
            (pl.col("LastPrice") / (pl.col("_prev_last") + eps)).log().fill_null(0.0).alias("_log_ret"),
            
            # 5. Lee-Ready 交易方向判定 D_t in {+1, -1, 0}
            pl.when(pl.col("_delta_v") == 0).then(pl.lit(0))
            .when(pl.col("LastPrice") >= pl.col("_prev_ask")).then(pl.lit(1))
            .when(pl.col("LastPrice") <= pl.col("_prev_bid")).then(pl.lit(-1))
            .when(pl.col("LastPrice") > pl.col("_prev_last")).then(pl.lit(1))
            .when(pl.col("LastPrice") < pl.col("_prev_last")).then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias("_trade_dir"),

            # 6. 一档 OFI 单跳增量 (Cont 2014)
            (
                pl.when(pl.col("BidPrice1") > pl.col("_prev_bid")).then(pl.col("BidVolume1"))
                .when(pl.col("BidPrice1") == pl.col("_prev_bid")).then(pl.col("BidVolume1") - pl.col("_prev_bid_vol"))
                .otherwise(-pl.col("_prev_bid_vol"))
                -
                pl.when(pl.col("AskPrice1") < pl.col("_prev_ask")).then(-pl.col("AskVolume1"))
                .when(pl.col("AskPrice1") == pl.col("_prev_ask")).then(pl.col("AskVolume1") - pl.col("_prev_ask_vol"))
                .otherwise(pl.col("_prev_ask_vol"))
            ).fill_null(0.0).alias("_ofi_tick"),

            # 7. 盘口击穿消耗量
            pl.when(pl.col("BidPrice1") < pl.col("_prev_bid")).then(pl.col("_prev_bid_vol")).otherwise(0.0).alias("_bid_deplete"),
            pl.when(pl.col("AskPrice1") > pl.col("_prev_ask")).then(pl.col("_prev_ask_vol")).otherwise(0.0).alias("_ask_deplete"),

            # 8. 深度失衡与单跳 VWAP
            ((pl.col("BidVolume1") - pl.col("AskVolume1")) / (pl.col("BidVolume1") + pl.col("AskVolume1") + eps)).alias("_depth_imb"),
            (pl.col("_delta_m") / (pl.col("_delta_v") * multiplier + eps)).alias("_vwap_tick"),
        ])
    )


def resample_to_1min(df_primitives: pl.LazyFrame, multiplier: float = 10.0) -> pl.LazyFrame:
    """
    阶段 3 & 4: 1 分钟切片聚合与全套指标比率化计算
    """
    eps = 1e-7

    return (
        df_primitives
        .with_columns(
            pl.col("timestamp").dt.truncate("1m").alias("trade_time"),
            pl.col("InstrumentID").alias("code"),
        )
        .group_by(["trade_time", "code"])
        .agg([
            # -------------------------------------------------------------
            # 1. 基础量价 (future_1min) 聚合
            # -------------------------------------------------------------
            pl.col("LastPrice").first().alias("open"),
            pl.col("LastPrice").max().alias("high"),
            pl.col("LastPrice").min().alias("low"),
            pl.col("LastPrice").last().alias("close"),
            pl.col("_delta_v").sum().alias("volume"),
            pl.col("_delta_m").sum().alias("money"),
            pl.col("LastPrice").mean().alias("twap"),
            pl.len().alias("tick_count"),

            # -------------------------------------------------------------
            # 2. 资金流向 (money_flow) 聚合
            # -------------------------------------------------------------
            (pl.col("_trade_dir") == 1).sum().alias("tick_in"),
            (pl.col("_trade_dir") == -1).sum().alias("tick_out"),
            pl.col("_delta_v").filter(pl.col("_trade_dir") == 1).sum().alias("volume_in"),
            pl.col("_delta_v").filter(pl.col("_trade_dir") == -1).sum().alias("volume_out"),
            pl.col("_delta_m").filter(pl.col("_trade_dir") == 1).sum().alias("money_in"),
            pl.col("_delta_m").filter(pl.col("_trade_dir") == -1).sum().alias("money_out"),

            # 聪明钱定义：单跳量超过本分钟 90% 分位数
            pl.col("_delta_v").filter((pl.col("_trade_dir") == 1) & (pl.col("_delta_v") >= pl.col("_delta_v").quantile(0.90))).sum().alias("smart_volume_in"),
            pl.col("_delta_v").filter((pl.col("_trade_dir") == -1) & (pl.col("_delta_v") >= pl.col("_delta_v").quantile(0.90))).sum().alias("smart_volume_out"),
            pl.col("_delta_m").filter((pl.col("_trade_dir") == 1) & (pl.col("_delta_v") >= pl.col("_delta_v").quantile(0.90))).sum().alias("smart_money_in"),
            pl.col("_delta_m").filter((pl.col("_trade_dir") == -1) & (pl.col("_delta_v") >= pl.col("_delta_v").quantile(0.90))).sum().alias("smart_money_out"),

            # -------------------------------------------------------------
            # 3. 盘口不平衡与波动 (imbalance) 聚合
            # -------------------------------------------------------------
            pl.col("_spread").mean().alias("bid_ask_spread"),
            pl.col("_spread").std().fill_null(0.0).alias("spread_std"),
            (pl.col("_spread") / (pl.col("_mid_price") + eps)).mean().alias("relative_spread"),
            pl.col("_depth_imb").mean().alias("depth_imbalance_1"),
            pl.col("_depth_imb").std().fill_null(0.0).alias("depth_imbalance_std"),
            pl.col("_depth_imb").last().alias("depth_imbalance_last"),
            ((pl.col("_micro_price") - pl.col("_mid_price")) / (pl.col("_mid_price") + eps)).mean().alias("micro_price_bias"),
            (pl.col("_log_ret").pow(2).sum()).sqrt().alias("realized_volatility"),
            ((pl.col("_log_ret").abs() * pl.col("_log_ret").abs().shift(1)).sum() * (math.pi / 2.0)).alias("realized_bipower_var"),

            # -------------------------------------------------------------
            # 4. 订单流 (order_flow) 聚合
            # -------------------------------------------------------------
            pl.col("_ofi_tick").sum().alias("ofi_sum"),
            pl.col("_ofi_tick").mean().alias("ofi_mean"),
            pl.col("_ofi_tick").std().fill_null(0.0).alias("ofi_std"),
            ((pl.col("BidVolume1") - pl.col("_prev_bid_vol")) - (pl.col("AskVolume1") - pl.col("_prev_ask_vol"))).sum().alias("voi_1"),
            pl.col("_bid_deplete").sum().alias("bid_depletion_rate"),
            pl.col("_ask_deplete").sum().alias("ask_depletion_rate"),

            # -------------------------------------------------------------
            # 5. 微观相关性 (corr) 聚合
            # -------------------------------------------------------------
            pl.corr("_delta_m", "_log_ret").fill_nan(0.0).fill_null(0.0).alias("corr_money_ret"),
            pl.corr("_delta_m", "_spread").fill_nan(0.0).fill_null(0.0).alias("corr_money_spread"),
            pl.corr("_ofi_tick", "_log_ret").fill_nan(0.0).fill_null(0.0).alias("corr_ofi_ret"),
            pl.corr("_vwap_tick", "_spread").fill_nan(0.0).fill_null(0.0).alias("corr_vwap_spread"),
            pl.corr("_delta_v", "_depth_imb").fill_nan(0.0).fill_null(0.0).alias("corr_vol_depth_imb"),

            # -------------------------------------------------------------
            # 6. 持仓博弈 (open_interest) 聚合
            # -------------------------------------------------------------
            (pl.col("OpenInterest").last() - pl.col("OpenInterest").first()).alias("delta_oi"),
            pl.col("_delta_oi").abs().sum().alias("abs_delta_oi"),
            pl.col("_delta_v").filter((pl.col("_delta_oi") > 0) & (pl.col("_delta_oi") >= 0.8 * pl.col("_delta_v"))).sum().alias("double_open_vol"),
            pl.col("_delta_v").filter((pl.col("_delta_oi") < 0) & (-pl.col("_delta_oi") >= 0.8 * pl.col("_delta_v"))).sum().alias("double_close_vol"),
            pl.col("_delta_v").filter(pl.col("_delta_oi").abs() < 0.2 * pl.col("_delta_v")).sum().alias("swap_volume"),
            pl.col("_delta_v").filter((pl.col("_trade_dir") == 1) & (pl.col("_delta_oi") > 0)).sum().alias("bull_active_open"),
            pl.col("_delta_v").filter((pl.col("_trade_dir") == -1) & (pl.col("_delta_oi") > 0)).sum().alias("bear_active_open"),
            pl.col("_delta_v").filter((pl.col("_trade_dir") == -1) & (pl.col("_delta_oi") < 0)).sum().alias("bull_stop_loss_vol"),
            pl.col("_delta_v").filter((pl.col("_trade_dir") == 1) & (pl.col("_delta_oi") < 0)).sum().alias("bear_stop_loss_vol"),
            ((pl.col("_delta_oi").abs() * pl.col("LastPrice")).sum() / (pl.col("_delta_oi").abs().sum() + eps)).alias("oi_weighted_price"),

            # -------------------------------------------------------------
            # 7 & 8. 均价锚点与涨跌停聚合
            # -------------------------------------------------------------
            pl.col("AveragePrice").first().alias("_avg_price_first"),
            pl.col("AveragePrice").last().alias("_avg_price_last"),
            (pl.col("LastPrice") >= pl.col("AveragePrice")).mean().alias("price_above_avg_time"),
            pl.col("UpperLimitPrice").last().alias("_upper_limit"),
            pl.col("LowerLimitPrice").last().alias("_lower_limit"),
            ((pl.col("BidVolume1") + pl.col("AskVolume1")).last() / (pl.col("_delta_v").last() + eps)).alias("near_limit_liquidity"),
        ])
        # -----------------------------------------------------------------
        # 阶段 4: 派生比率、百分比与除零保护 (全套无量纲绿灯特征生成)
        # -----------------------------------------------------------------
        .with_columns([
            # 基础派生
            (pl.col("money") / (pl.col("volume") * multiplier + eps)).alias("vwap"),
            ((pl.col("close") - pl.col("open")) / (pl.col("open") + eps)).alias("pct_change"),
            ((pl.col("high") - pl.col("low")) / (pl.col("open") + eps)).alias("high_low_ratio"),
            (pl.col("volume") / (pl.col("tick_count") + eps)).alias("volume_per_tick"),
            
            # 资金流比率
            (pl.col("tick_in") - pl.col("tick_out")).alias("net_tick_in"),
            (pl.col("volume_in") - pl.col("volume_out")).alias("net_volume_in"),
            (pl.col("money_in") - pl.col("money_out")).alias("net_money_in"),
            (pl.col("tick_in") / (pl.col("tick_count") + eps)).alias("tick_in_pct"),
            (pl.col("tick_out") / (pl.col("tick_count") + eps)).alias("tick_out_pct"),
            ((pl.col("tick_in") - pl.col("tick_out")) / (pl.col("tick_count") + eps)).alias("net_tick_in_pct"),
            (pl.col("volume_in") / (pl.col("volume") + eps)).alias("volume_in_pct"),
            (pl.col("volume_out") / (pl.col("volume") + eps)).alias("volume_out_pct"),
            ((pl.col("volume_in") - pl.col("volume_out")) / (pl.col("volume") + eps)).alias("net_volume_in_pct"),
            ((pl.col("money_in") - pl.col("money_out")) / (pl.col("money") + eps)).alias("net_money_in_pct"),
            (pl.col("smart_volume_in") / (pl.col("volume") + eps)).alias("smart_volume_in_pct"),
            (pl.col("smart_volume_out") / (pl.col("volume") + eps)).alias("smart_volume_out_pct"),
            (pl.col("smart_money_in") / (pl.col("money") + eps)).alias("smart_money_in_pct"),
            (pl.col("smart_money_out") / (pl.col("money") + eps)).alias("smart_money_out_pct"),
            ((pl.col("smart_volume_in") - pl.col("smart_volume_out")) / (pl.col("volume") + eps)).alias("smart_net_vol_pct"),

            # 波动跳跃率
            (
                pl.when(pl.col("realized_volatility") > 0)
                .then((pl.col("realized_volatility").pow(2) - pl.col("realized_bipower_var")).clip(lower_bound=0) / (pl.col("realized_volatility").pow(2) + eps))
                .otherwise(0.0)
            ).alias("jump_ratio"),

            # 订单流比率
            (pl.col("ofi_sum") / (pl.col("volume") + eps)).alias("ofi_normalized"),
            (pl.col("voi_1") / (pl.col("volume") + eps)).alias("voi_normalized"),
            (pl.col("bid_depletion_rate") / (pl.col("volume") + eps)).alias("bid_depletion_ratio"),
            (pl.col("ask_depletion_rate") / (pl.col("volume") + eps)).alias("ask_depletion_ratio"),
            ((pl.col("bid_depletion_rate") - pl.col("ask_depletion_rate")) / (pl.col("volume") + eps)).alias("depletion_imbalance"),

            # 持仓博弈比率
            (pl.col("delta_oi") / (pl.col("volume") + eps)).alias("delta_oi_ratio"),
            (pl.col("abs_delta_oi") / (pl.col("volume") + eps)).alias("abs_delta_oi_ratio"),
            (pl.col("delta_oi").abs() / (pl.col("volume") + eps)).alias("oi_volume_ratio"),
            (pl.col("double_open_vol") / (pl.col("volume") + eps)).alias("double_open_ratio"),
            (pl.col("double_close_vol") / (pl.col("volume") + eps)).alias("double_close_ratio"),
            (pl.col("swap_volume") / (pl.col("volume") + eps)).alias("swap_volume_ratio"),
            (pl.col("bull_active_open") / (pl.col("volume") + eps)).alias("bull_active_open_ratio"),
            (pl.col("bear_active_open") / (pl.col("volume") + eps)).alias("bear_active_open_ratio"),
            (pl.col("bull_stop_loss_vol") / (pl.col("volume") + eps)).alias("bull_stop_loss_ratio"),
            (pl.col("bear_stop_loss_vol") / (pl.col("volume") + eps)).alias("bear_stop_loss_ratio"),
            ((pl.col("bull_active_open") - pl.col("bear_active_open")) / (pl.col("volume") + eps)).alias("oi_flow_imbalance"),

            # 均价锚点比率
            ((pl.col("close") - pl.col("_avg_price_last")) / (pl.col("_avg_price_last") + eps)).alias("price_to_avg_dev"),
            ((pl.col("_avg_price_last") - pl.col("_avg_price_first")) / (pl.col("_avg_price_first") + eps)).alias("avg_price_slope"),

            # 涨跌停距离与不对称性
            ((pl.col("_upper_limit") - pl.col("close")) / (pl.col("close") + eps)).alias("dist_upper_limit"),
            ((pl.col("close") - pl.col("_lower_limit")) / (pl.col("close") + eps)).alias("dist_lower_limit"),
        ])
        .with_columns([
            # 依赖上面产生列的二级复合列
            ((pl.col("close") - pl.col("vwap")) / (pl.col("vwap") + eps)).alias("vwap_close_bias"),
            ((pl.col("vwap") - pl.col("_avg_price_last")) / (pl.col("_avg_price_last") + eps)).alias("vwap_to_avg_bias"),
            (
                (pl.col("dist_upper_limit") - pl.col("dist_lower_limit"))
                / (pl.col("dist_upper_limit") + pl.col("dist_lower_limit") + eps)
            ).alias("limit_bound_asymmetry"),
        ])
        # 移除下划线打头的临时列
        .select(pl.exclude(r"^_\w+$"))
        .sort(["trade_time", "code"])
    )


def compute(df_lazy: pl.LazyFrame, multiplier: float = 10.0) -> pl.LazyFrame:
    """
    全量特征计算入口：接收原始 CTP Tick LazyFrame，返回 1 分钟全量宽表
    """
    primitives = build_tick_primitives(df_lazy, multiplier=multiplier)
    return resample_to_1min(primitives, multiplier=multiplier)
