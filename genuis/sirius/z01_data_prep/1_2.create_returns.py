## Crypto BN 收益率计算 (Polars 版本)
import os, datetime,pdb
import polars as pl
from pathlib import Path
import pyarrow.feather as pf
from dotenv import load_dotenv

load_dotenv()

#from kdutils.ttimes import get_dates
#from kdutils.macro2 import TASK_MAPPING, base_path
#from kdutils.tactix import Tactix


def create_long_perp_return(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    做多永续合约单步收益 (open-to-open, log-return)

    数学定义 (文档 §2.1):
        R_long_perp = R_future - Funding
        y_long_perp = log(1 + R_long_perp)

    Funding > 0 → 多头支付给空头 → 多头收益减少
    Funding < 0 → 空头支付给多头 → 多头收益增加

    输入须含列: trade_time, code, f_open, f_funding_rate
    """
    return (
        df_lazy
        .sort(["trade_time", "code"])
        .with_columns(
            # 永续合约单步 simple return: R_future = (F_{t+1} - F_t) / F_t
            pl.col("f_open").pct_change().over("code").alias("_f_ret")
        )
        .with_columns([
            # 对齐：让 T 行代表 [T+1 open -> T+2 open) 的收益
            pl.col("_f_ret").shift(-2).over("code").alias("_f_ret_aligned"),
            # T 行用到 f_{T+1}（对应 [T+1, T+2) 区间的 funding）
            pl.col("f_funding_rate").shift(-1).over("code").alias("_fund_aligned"),
        ])
        .with_columns(
            # 做多永续：y_long = log(1 + R_future - Funding)
            (1 + pl.col("_f_ret_aligned") - pl.col("_fund_aligned"))
            .log().alias("long_perp_ret")
        )
        .select(["trade_time", "code", "long_perp_ret"])
    )


def create_short_perp_return(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    做空永续合约单步收益 (open-to-open, log-return)

    数学定义 (文档 §2.1):
        R_short_perp = -R_future + Funding
        y_short_perp = log(1 + R_short_perp)

    Funding > 0 → 空头收到支付 → 空头收益增加
    Funding < 0 → 空头支付给多头 → 空头收益减少

    ⚠️ 不可通过 -long_perp_ret 得到，funding 在 log 域非线性，必须独立计算。

    输入须含列: trade_time, code, f_open, f_funding_rate
    """
    return (
        df_lazy
        .sort(["trade_time", "code"])
        .with_columns(
            pl.col("f_open").pct_change().over("code").alias("_f_ret")
        )
        .with_columns([
            pl.col("_f_ret").shift(-2).over("code").alias("_f_ret_aligned"),
            pl.col("f_funding_rate").shift(-1).over("code").alias("_fund_aligned"),
        ])
        .with_columns(
            # 做空永续：y_short = log(1 - R_future + Funding)
            (1 - pl.col("_f_ret_aligned") + pl.col("_fund_aligned"))
            .log().alias("short_perp_ret")
        )
        .select(["trade_time", "code", "short_perp_ret"])
    )


def create_chg_directional(df_lazy: pl.LazyFrame, cost_rate: float = 0.0) -> pl.LazyFrame:
    """
    计算单向永续合约的单步方向性收益 (log-return)，扣除交易成本。

    在一次 pass 中同时输出做多和做空的 log-return。
    符合文档 §2.1 定义:
        R_long_perp  = R_future - Funding
        R_short_perp = -R_future + Funding

    输入须含列: trade_time, code, f_open, f_funding_rate
    输出列:     trade_time, code, long_perp_ret, short_perp_ret
    """
    return (
        df_lazy
        .sort(["trade_time", "code"])
        .with_columns(
            pl.col("f_open").pct_change().over("code").alias("_f_ret")
        )
        .with_columns([
            pl.col("_f_ret").shift(-2).over("code").alias("_f_ret_aligned"),
            pl.col("f_funding_rate").shift(-1).over("code").alias("_fund_aligned"),
        ])
        .with_columns([
            # 在 simple return 域扣除成本后转 log
            (1 + pl.col("_f_ret_aligned") - pl.col("_fund_aligned") - cost_rate)
            .log().alias("long_perp_ret"),
            (1 - pl.col("_f_ret_aligned") + pl.col("_fund_aligned") - cost_rate)
            .log().alias("short_perp_ret"),
        ])
        .select(["trade_time", "code", "long_perp_ret", "short_perp_ret"])
    )


def create_return(df_lazy: pl.LazyFrame, cost_rate: float = 0.0) -> pl.LazyFrame:
    """
    计算多 horizon 的方向性永续合约收益

    输出列:
        long_perp_ret_1h ~ long_perp_ret_15h
        short_perp_ret_1h ~ short_perp_ret_15h
        long_time_weight, long_equal_weight
        short_time_weight, short_equal_weight
    """
    horizon_sets = [1, 2, 3, 5, 10, 15]

    ## Step 1: 计算单步方向性收益
    chg = create_chg_directional(df_lazy, cost_rate=cost_rate)

    ## Step 2: 对 long_perp_ret 和 short_perp_ret 分别做多 horizon 滚动
    horizon_exprs = []
    for horizon in horizon_sets:
        for col in ["long_perp_ret", "short_perp_ret"]:
            horizon_exprs.append(
                pl.col(col)
                .rolling_sum(window_size=horizon, min_periods=1)
                .shift(-(horizon - 1))
                .over("code")
                .alias(f"{col}_{horizon}h")
            )

    result = chg.with_columns(horizon_exprs)

    ## Step 3: 加权混合（分别对多头和空头独立计算）
    w1, w2, w3 = 3 / 6, 2 / 6, 1 / 6  # T+1 权重最大, T+2 其次, T+3 最小

    result = result.with_columns([
        # 多头加权
        (pl.col("long_perp_ret_1h") * w1
         + pl.col("long_perp_ret_2h") * w2
         + pl.col("long_perp_ret_3h") * w3).alias("long_time_weight"),
        ((pl.col("long_perp_ret_1h") + pl.col("long_perp_ret_2h")
          + pl.col("long_perp_ret_3h")) / 3).alias("long_equal_weight"),
        # 空头加权
        (pl.col("short_perp_ret_1h") * w1
         + pl.col("short_perp_ret_2h") * w2
         + pl.col("short_perp_ret_3h") * w3).alias("short_time_weight"),
        ((pl.col("short_perp_ret_1h") + pl.col("short_perp_ret_2h")
          + pl.col("short_perp_ret_3h")) / 3).alias("short_equal_weight"),
    ])

    ## 选择输出列
    out_cols = ["trade_time", "code"]
    for h in horizon_sets:
        out_cols.extend([f"long_perp_ret_{h}h", f"short_perp_ret_{h}h"])
    out_cols.extend([
        "long_time_weight", "long_equal_weight",
        "short_time_weight", "short_equal_weight",
    ])

    return result.select(out_cols)


def returns_save(return_data: pl.LazyFrame, method: str, task_id: str):
    start_date, end_date = get_dates(method)
    start_dt = (datetime.datetime.strptime(start_date, '%Y-%m-%d')
                + datetime.timedelta(days=1))
    end_dt = (datetime.datetime.strptime(end_date, '%Y-%m-%d')
              + datetime.timedelta(days=1))

    df = (
        return_data
        .filter(
            (pl.col("trade_time") >= start_dt)
            & (pl.col("trade_time") <= end_dt)
        )
        .sort(["trade_time", "code"])
        .collect()
    )

    dirs = os.path.join(base_path, method, 'derivative', task_id)
    os.makedirs(dirs, exist_ok=True)
    filename = os.path.join(dirs, 'returns_data.feather')
    print(filename)
    df.write_ipc(filename)


def run(method, task_id):
    #dirs = os.path.join(base_path, method, 'basic', task_id)
    #file_name = os.path.join(dirs, "raw_basic.feather")
    pdb.set_trace()
    file_name = "/workspace/worker/pj/Chrono/genuis/orion/records/cicso0/basic/1000201201/raw_basic.feather"
    raw_data = pl.from_arrow(pf.read_table(file_name)).lazy()

    # 方向性收益只需 f_open 和 f_funding_rate
    data = raw_data.select(['trade_time', 'code', 'f_open', 'f_funding_rate'])

    return_data = create_return(data, cost_rate=0.0).drop_nulls()
    print(return_data.collect())
    #returns_save(return_data=return_data, task_id=task_id, method=method)


if __name__ == '__main__':
    #variant = Tactix().start()
    run(method='1', task_id='2')
