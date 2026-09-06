import polars as pl


def preprocess_ticks(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    schema_names = df_lazy.collect_schema().names()

    # 阶段 1: timestamp 标准化
    if "TickDatetime" in schema_names:
        df_tick = df_lazy.with_columns(
            pl.col("TickDatetime").alias("timestamp"))
    elif "timestamp" not in schema_names:
        df_tick = df_lazy.with_columns(
            pl.concat_str([
                pl.col("TradingDay").cast(pl.Utf8),
                pl.lit(" "),
                pl.col("UpdateTime").cast(pl.Utf8),
                pl.lit("."),
                pl.col("UpdateMillisec").cast(pl.Utf8).str.zfill(3),
            ])
            .str.strptime(pl.Datetime, format="%Y%m%d %H:%M:%S.%3f", strict=False)
            .alias("timestamp")
        )
    else:
        df_tick = df_lazy

    # 标的与品种代码标准化: symbol = 合约代码 (如 rb1205), code = 品种大写 (如 RB)
    contract_col = None
    if "symbol" in schema_names:
        contract_col = "symbol"
    elif "InstrumentID" in schema_names:
        contract_col = "InstrumentID"
    elif "code" in schema_names:
        contract_col = "code"

    norm_cols = []
    # 保证 symbol 列存在 (完整合约代码，如 rb1205)
    if "symbol" not in schema_names:
        if contract_col is not None:
            norm_cols.append(pl.col(contract_col).cast(pl.Utf8).alias("symbol"))
        else:
            raise ValueError("输入数据必须包含 symbol, InstrumentID 或 code 之一")

    # 保证 code 列为纯英文字母大写的品种代码 (如 RB)
    if contract_col is not None:
        norm_cols.append(
            pl.col(contract_col).cast(pl.Utf8).str.extract(r"^([a-zA-Z]+)", 1).str.to_uppercase().alias("code")
        )

    if norm_cols:
        df_tick = df_tick.with_columns(norm_cols)

    group_col = "symbol"

    res = (
        df_tick
        .sort([group_col, "timestamp"])
        .with_columns([
            pl.col("Volume").diff().clip(lower_bound=0).fill_null(
                0).over(group_col).alias("_delta_v"),
            pl.col("Turnover").diff().clip(lower_bound=0).fill_null(
                0.0).over(group_col).alias("_delta_m"),
            pl.col("LastPrice").shift(1).over(
                group_col).alias("_prev_last"),
            pl.col("AskPrice1").shift(1).over(
                group_col).alias("_prev_ask"),
            pl.col("BidPrice1").shift(1).over(
                group_col).alias("_prev_bid"),
        ])
        .with_columns([
            pl.when(pl.col("_delta_v") == 0).then(pl.lit(0))
            .when(pl.col("LastPrice") >= pl.col("_prev_ask")).then(pl.lit(1))
            .when(pl.col("LastPrice") <= pl.col("_prev_bid")).then(pl.lit(-1))
            .when(pl.col("LastPrice") > pl.col("_prev_last")).then(pl.lit(1))
            .when(pl.col("LastPrice") < pl.col("_prev_last")).then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias("_trade_dir"),
        ])
    )

    if "AskPrice1" in schema_names and "BidPrice1" in schema_names:
        res = res.with_columns([
            (pl.col("AskPrice1") - pl.col("BidPrice1")).alias("_spread"),
            ((pl.col("AskPrice1") + pl.col("BidPrice1")) / 2.0).alias("_mid_price"),
            pl.when((pl.col("LastPrice") > 0) & (pl.col("_prev_last") > 0))
            .then((pl.col("LastPrice") / pl.col("_prev_last")).log())
            .otherwise(0.0)
            .alias("_log_ret"),
        ]).with_columns([
            pl.when(pl.col("_mid_price") > 0)
            .then(pl.col("_spread") / pl.col("_mid_price"))
            .otherwise(0.0)
            .alias("_rel_spread"),
            pl.col("_log_ret").pow(2).alias("_sq_log_ret"),
            pl.col("_log_ret").abs().alias("_abs_log_ret"),
        ]).with_columns([
            (pl.col("_abs_log_ret") * pl.col("_abs_log_ret").shift(1).over(group_col)).fill_null(0.0).alias("_bipower_prod"),
        ])

    if "AskVolume1" in schema_names and "BidVolume1" in schema_names:
        res = res.with_columns([
            (pl.col("BidVolume1") + pl.col("AskVolume1")).alias("_vol_sum"),
        ]).with_columns([
            pl.when(pl.col("_vol_sum") > 0)
            .then((pl.col("BidVolume1") - pl.col("AskVolume1")) / pl.col("_vol_sum"))
            .otherwise(0.0)
            .alias("_depth_imb"),
            pl.when((pl.col("_mid_price") > 0) & (pl.col("_vol_sum") > 0))
            .then(
                ((pl.col("AskPrice1") * pl.col("BidVolume1") + pl.col("BidPrice1") * pl.col("AskVolume1")) / pl.col("_vol_sum") - pl.col("_mid_price"))
                / pl.col("_mid_price")
            )
            .otherwise(0.0)
            .alias("_micro_bias"),
            pl.col("BidVolume1").shift(1).over(group_col).alias("_prev_bid_v"),
            pl.col("AskVolume1").shift(1).over(group_col).alias("_prev_ask_v"),
        ]).with_columns([
            (
                pl.when(pl.col("BidPrice1") > pl.col("_prev_bid")).then(pl.col("BidVolume1"))
                .when(pl.col("BidPrice1") == pl.col("_prev_bid")).then(pl.col("BidVolume1") - pl.col("_prev_bid_v"))
                .when(pl.col("BidPrice1") < pl.col("_prev_bid")).then(-pl.col("_prev_bid_v"))
                .otherwise(0.0)
                - (
                    pl.when(pl.col("AskPrice1") < pl.col("_prev_ask")).then(pl.col("AskVolume1"))
                    .when(pl.col("AskPrice1") == pl.col("_prev_ask")).then(pl.col("AskVolume1") - pl.col("_prev_ask_v"))
                    .when(pl.col("AskPrice1") > pl.col("_prev_ask")).then(-pl.col("_prev_ask_v"))
                    .otherwise(0.0)
                )
            ).fill_null(0.0).alias("_ofi"),
            (
                (pl.col("BidVolume1") - pl.col("_prev_bid_v")) - (pl.col("AskVolume1") - pl.col("_prev_ask_v"))
            ).fill_null(0.0).alias("_voi"),
            pl.when(pl.col("BidPrice1") < pl.col("_prev_bid")).then(pl.col("_prev_bid_v")).otherwise(0.0).fill_null(0.0).alias("_bid_depletion"),
            pl.when(pl.col("AskPrice1") > pl.col("_prev_ask")).then(pl.col("_prev_ask_v")).otherwise(0.0).fill_null(0.0).alias("_ask_depletion"),
            pl.when(pl.col("_delta_v") > 0).then(pl.col("_delta_m") / pl.col("_delta_v")).otherwise(pl.col("LastPrice")).fill_null(0.0).alias("_tick_vwap"),
        ])

    if "OpenInterest" in schema_names:
        res = res.with_columns([
            pl.col("OpenInterest").diff().fill_null(0).over(group_col).alias("_delta_oi"),
        ]).with_columns([
            pl.col("_delta_oi").abs().alias("_abs_delta_oi"),
            pl.when((pl.col("_delta_oi") > 0) & (pl.col("_delta_oi") >= 0.8 * pl.col("_delta_v")))
            .then(pl.col("_delta_v")).otherwise(0).alias("_double_open_vol"),
            pl.when((pl.col("_delta_oi") < 0) & (-pl.col("_delta_oi") >= 0.8 * pl.col("_delta_v")))
            .then(pl.col("_delta_v")).otherwise(0).alias("_double_close_vol"),
            pl.when(pl.col("_delta_oi").abs() < 0.2 * pl.col("_delta_v"))
            .then(pl.col("_delta_v")).otherwise(0).alias("_swap_vol"),
            pl.when((pl.col("_trade_dir") == 1) & (pl.col("_delta_oi") > 0))
            .then(pl.col("_delta_v")).otherwise(0).alias("_bull_active_open"),
            pl.when((pl.col("_trade_dir") == -1) & (pl.col("_delta_oi") > 0))
            .then(pl.col("_delta_v")).otherwise(0).alias("_bear_active_open"),
            pl.when((pl.col("_trade_dir") == -1) & (pl.col("_delta_oi") < 0))
            .then(pl.col("_delta_v")).otherwise(0).alias("_bull_stop_loss"),
            pl.when((pl.col("_trade_dir") == 1) & (pl.col("_delta_oi") < 0))
            .then(pl.col("_delta_v")).otherwise(0).alias("_bear_stop_loss"),
        ]).with_columns([
            (pl.col("_abs_delta_oi") * pl.col("LastPrice")).alias("_oi_price_prod"),
        ])

    if "AveragePrice" in schema_names:
        res = res.with_columns([
            pl.when(pl.col("LastPrice") >= pl.col("AveragePrice")).then(1.0).otherwise(0.0).alias("_above_avg"),
            (pl.col("_delta_v") * pl.col("LastPrice")).alias("_vol_price_prod"),
        ])

    if "UpperLimitPrice" in schema_names and "LowerLimitPrice" in schema_names:
        res = res.with_columns([
            ((pl.col("UpperLimitPrice") - pl.col("LastPrice")) / (pl.col("LastPrice") + 1e-7)).alias("_dist_up"),
            ((pl.col("LastPrice") - pl.col("LowerLimitPrice")) / (pl.col("LastPrice") + 1e-7)).alias("_dist_down"),
        ]).with_columns([
            pl.min_horizontal("_dist_up", "_dist_down").alias("_min_dist_limit"),
        ]).with_columns([
            pl.when(pl.col("_min_dist_limit") <= 0.005)
            .then((pl.col("BidVolume1") + pl.col("AskVolume1")) / (pl.col("_delta_v") + 1e-7))
            .otherwise(None)
            .alias("_near_limit_liq"),
        ])

    return res

