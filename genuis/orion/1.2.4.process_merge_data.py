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

funding_rename_map = {
    "last_funding_rate": "funding_rate",
    "funding_interval_hours": "funding_interval"
}

um_rename_map = {
    "volume": "volume",
    "quote_volume": "value",
    "taker_buy_volume": "buy_vol",
    "taker_buy_quote_volume": "buy_value"
}


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

    return df


# --- 主加载逻辑 ---
def load_and_merge_data(method, task_id):
    base_dirs = os.path.join(base_path, method, "basic", task_id)
    print(f"Loading data from: {base_dirs}")
    pdb.set_trace()

    # 1. 加载数据
    try:
        um_kline = pd.read_feather(os.path.join(base_dirs, "kline_um.feather"))
        um_metrics = pd.read_feather(
            os.path.join(base_dirs, "metrics_um.feather"))
        um_funding = pd.read_feather(
            os.path.join(base_dirs, "funding_um.feather"))
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None

    um_kline = um_kline.rename(columns=um_rename_map)
    if 'close_time' in um_kline.columns:
        um_kline = um_kline.drop(columns=['close_time'])
    um_kline['trade_time'] = pd.to_datetime(
        um_kline['trade_time']).dt.floor('s')

    um_funding = preprocess_dataframe(um_funding,
                                      funding_rename_map,
                                      filter_cols=True)
    um_funding = um_funding.drop_duplicates(subset=['trade_time', 'code'])
    final_df = pd.merge(um_metrics,
                        um_kline,
                        on=['trade_time', 'code'],
                        how='inner')
    final_df = pd.merge(final_df,
                        um_funding,
                        on=['trade_time', 'code'],
                        how='left')
    final_df['funding_rate'] = final_df['funding_rate'].fillna(0)
    final_df = final_df.drop(['funding_interval'], axis=1)

    ## 通过流通性过滤 # # 条件：日成交额 > 500w 且 持仓价值 > 200w 且 每小时成交笔数 > 500
    min_daily_value = 5000000
    min_oi_value = 2000000
    min_avg_count = 500
    stats = final_df.groupby('symbol').agg({
        'value': 'mean',
        'sum_open_interest_value': 'mean',
        'count': 'mean'
    })
    stats['daily_value_est'] = stats['value'] * 24
    valid_symbols = stats[(stats['daily_value_est'] >= min_daily_value)
                          & (stats['sum_open_interest_value'] >= min_oi_value)
                          & (stats['count'] >= min_avg_count)].index.tolist()
    final_df1 = final_df[final_df['symbol'].isin(valid_symbols)].copy()
    print(f"Merge complete. Shape: {final_df.shape}")
    return final_df1.sort_values(by=['trade_time','code']).reset_index(drop=True)


def start(method, task_id):
    # 获取并返回数据
    raw_basic_data = load_and_merge_data(method=method, task_id=task_id)
    output_dirs = os.path.join(base_path, method, "basic", task_id)
    os.makedirs(output_dirs, exist_ok=True)
    filename = os.path.join(output_dirs, f"raw_basic.feather")
    raw_basic_data.to_feather(filename)


if __name__ == '__main__':
    variant = Tactix().start()
    final_data = start(method=variant.method, task_id=variant.task_id)
