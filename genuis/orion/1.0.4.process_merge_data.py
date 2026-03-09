## BN 数据合并
import pdb, os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 假设 base_path 从环境变量获取，或者你需要在这里定义
base_path = os.getenv("BASE_PATH", "/workspace/data")

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from kdutils.ttimes import get_dates

# --- 映射配置 ---
spot_rename_map = {
    "open": "s_open",
    "high": "s_high",
    "low": "s_low",
    "close": "s_close",
    "volume": "s_vol",
    "quote_volume": "s_value",
    "count": "s_cnt",
    "taker_buy_volume": "s_buy_vol",
    "taker_buy_quote_volume": "s_buy_value"
}

um_rename_map = {
    "open": "f_open",
    "high": "f_high",
    "low": "f_low",
    "close": "f_close",
    "volume": "f_vol",
    "quote_volume": "f_value",
    "count": "f_cnt",
    "taker_buy_volume": "f_buy_vol",
    "taker_buy_quote_volume": "f_buy_value"
}

metrics_rename_map = {
    "sum_open_interest": "f_oi",
    "sum_open_interest_value": "f_oi_value",
    "count_toptrader_long_short_ratio": "f_lsr_top_acc",
    "sum_toptrader_long_short_ratio": "f_lsr_top_pos",
    "count_long_short_ratio": "f_lsr_global",
    "sum_taker_long_short_vol_ratio": "f_lsr_taker"
}

funding_rename_map = {
    "last_funding_rate": "f_funding_rate",
    "funding_interval_hours": "f_funding_interval"
}

# --- 通用工具函数 ---


def normalize_codes(df):
    """
    通用代码归一化函数：去除 1000/1000000 前缀并处理重命名
    """
    # 1. 字符串替换
    df['code'] = df['code'].str.replace('1000000', '', regex=False)
    df['code'] = df['code'].str.replace('1000', '', regex=False)

    # 2. 特殊映射
    rename_rules = {
        "BEAMXUSDT": "BEAMUSDT",
        "1MBABYDOGEUSDT": "BABYDOGEUSDT",
        #"NEIROETHUSDT": "NEIROUSDT"
    }
    df['code'] = df['code'].replace(rename_rules)
    return df


def preprocess_dataframe(df, rename_map=None, filter_cols=False):
    """
    通用预处理：重命名 -> 时间对齐 -> 代码归一化
    """
    # 1. 重命名
    if rename_map:
        df = df.rename(columns=rename_map)

    # 2. 筛选列 (如果是 Metrics 或 Funding，只保留相关列)
    if filter_cols and rename_map:
        cols_to_keep = ['trade_time', 'code'] + list(rename_map.values())
        # 这里的 intersection 处理是为了防止 map 里有 key 但 df 里没有的列导致报错
        existing_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[existing_cols]

    # 3. 强制时间对齐 (只做一次)
    if not pd.api.types.is_datetime64_any_dtype(df['trade_time']):
        df['trade_time'] = pd.to_datetime(df['trade_time'])
    df['trade_time'] = df['trade_time'].dt.floor('s')

    # 4. 代码归一化 (如果是 UM 数据)
    # 简单判断：如果已经没有 'close_time' 且是 Spot 数据可能不需要，
    # 但为了保险，建议对 UM 数据显式调用。此处为了通用，假设所有输入都需要归一化。
    # 如果 Spot 数据不需要归一化，可以在外部控制。
    df = normalize_codes(df)

    return df


# --- 主加载逻辑 ---
def load_and_merge_data(method, task_id):
    base_dirs = os.path.join(base_path, method, "basic", task_id)
    print(f"Loading data from: {base_dirs}")
    pdb.set_trace()
    # 1. 加载数据
    try:
        um_kline = pd.read_feather(os.path.join(base_dirs, "kline_um.feather"))
        spot_kline = pd.read_feather(
            os.path.join(base_dirs, "kline_spot.feather"))
        um_metrics = pd.read_feather(
            os.path.join(base_dirs, "metrics_um.feather"))
        um_funding = pd.read_feather(
            os.path.join(base_dirs, "funding_um.feather"))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None

    # 2. 处理 Spot Kline (Spot 不需要去前缀，但需要对齐时间)
    # 注意：Spot 代码本身不需要归一化，所以这里单独写
    pdb.set_trace()
    spot_kline = spot_kline.rename(columns=spot_rename_map)
    if 'close_time' in spot_kline.columns:
        spot_kline = spot_kline.drop(columns=['close_time'])
    spot_kline['trade_time'] = pd.to_datetime(
        spot_kline['trade_time']).dt.floor('s')

    # 3. 处理 UM Kline
    um_kline = um_kline.rename(columns=um_rename_map)
    if 'close_time' in um_kline.columns:
        um_kline = um_kline.drop(columns=['close_time'])
    um_kline['trade_time'] = pd.to_datetime(
        um_kline['trade_time']).dt.floor('s')
    um_kline = normalize_codes(um_kline)  # 仅 UM 需要归一化

    # 4. 处理 Metrics
    um_metrics = preprocess_dataframe(um_metrics,
                                      metrics_rename_map,
                                      filter_cols=True)

    # 5. 处理 Funding (额外去重)
    um_funding = preprocess_dataframe(um_funding,
                                      funding_rename_map,
                                      filter_cols=True)
    um_funding = um_funding.drop_duplicates(subset=['trade_time', 'code'])

    print("Data loaded and preprocessed. Starting merge...")

    # 6. 合并数据 (Merge Sequence)
    # Step A: 合并 K 线 (Inner Join, 仅保留两者都有的)
    final_df = pd.merge(spot_kline,
                        um_kline,
                        on=['trade_time', 'code'],
                        how='inner')

    # Step B: 合并 Metrics (Left Join, 以 K 线为主)
    final_df = pd.merge(final_df,
                        um_metrics,
                        on=['trade_time', 'code'],
                        how='left')

    # Step C: 合并 Funding (Left Join)
    final_df = pd.merge(final_df,
                        um_funding,
                        on=['trade_time', 'code'],
                        how='left')
    final_df['f_funding_rate'] = final_df['f_funding_rate'].fillna(0)
    final_df = final_df.drop(['f_funding_interval'],axis=1)
    pdb.set_trace()
    print(f"Merge complete. Shape: {final_df.shape}")
    return final_df


def start(method, task_id):
    # 获取并返回数据
    raw_basic_data = load_and_merge_data(method=method, task_id=task_id)
    ## 保存
    pdb.set_trace()
    output_dirs = os.path.join(base_path, method, "basic", task_id)
    os.makedirs(output_dirs, exist_ok=True)
    filename = os.path.join(output_dirs, f"raw_basic.feather")
    raw_basic_data.to_feather(filename)


if __name__ == '__main__':
    variant = Tactix().start()
    final_data = start(method=variant.method,
                       task_id=variant.task_id)
