REGIME_FEATURES = [
    # 宏观周期
    "m1m2_diff_mom_3m",
    "pmi_prev",
    "credit_prev",
    "ppi_prev",

    # 全球流动性、汇率、利率和估值
    "spread_percentile_252d",
    "usd_cny_mom_20d",
    "cn_10y_percentile_5y",
    "pe_500_percentile",

    # 市场流动性和情绪环境
    "turnover_percentile_252",
    "up_down_ratio_pct_252",
    "breadth_percentile_252",

    # 波动率期限结构
    "rv_5",
    "rv_25",
    "rv_75",

    # 价格位置
    "position_box",

    # 期现、风格和基本面
    "basis_percentile_252",
    "size_style_spread",
    "yoy_momentum_60",
]

PREDICT_FEATURES = [
    # Smart Money
    "main_flow_ratio",
    "main_flow_acceleration",
    "oi_change_3d",

    # Sentiment Shock
    "price_ret_5d",
    "price_acc_5d",
    "basis_diff_3d",
    "basis_div_strength",

    # Mean Reversion
    "breadth_div_intensity_pos",
    "exhaustion_intensity",
    "panic_reversal_score",

    # Trend Momentum
    "momentum_vol_adj_20",
    "gap_continuation",
    "style_momentum_slope",
]

TEXT_FEATURES = [
    "domestic_macro",
    "external_shock",
    "systemic_risk",
    "domestic_policy",
    "global_liquidity",
]