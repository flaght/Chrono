import polars as pl
from feature.utils import formula as ops

file_path = (
    "/workspace/worker/pj/Chrono/genuis/mizar/records/"
    "ricso2/rbb/basic/recent_data.feather"
)

df_lazy = pl.scan_ipc(file_path).sort(["trade_time", "code"])

# 第一阶段：生成基础派生特征
factor_lazy = df_lazy.with_columns(
    ops.log_return("close").alias("_return"),
    ops.pct_change("volume").alias("_volume_change"),
    ops.pct_change("openint").alias("_openint_change"),
)

# 第二阶段：基于第一阶段结果生成滚动特征
factor_lazy = factor_lazy.with_columns(
    ops.ts_mean("_return", 10).alias("return_mean_10"),
    ops.ts_std("_return", 10).alias("return_std_10"),
    ops.ts_rank("_volume_change", 20).alias("volume_change_rank_20"),
    ops.ts_corr(
        "_return",
        "_openint_change",
        20,
    ).alias("return_openint_corr_20"),
)


# 第三阶段：组合已有结果
factor_lazy = factor_lazy.with_columns(
    ops.safe_div(
        "return_mean_10",
        "return_std_10",
    ).alias("return_mean_std_ratio")
)

result = (
    factor_lazy
    .select(
        [
            "trade_time",
            "code",
            "return_mean_10",
            "return_std_10",
            "volume_change_rank_20",
            "return_openint_corr_20",
            "return_mean_std_ratio",
        ]
    )
    .collect()
)

print(result)