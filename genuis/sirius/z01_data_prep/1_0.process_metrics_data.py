# BN处理 metrics 数据
import os, pdb
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from lib.utils.macro import *
from lib.utils.tactix import Tactix
from lib.utils.ttimes import get_dates

weighted_cols_map = {
    'count_toptrader_long_short_ratio': 'sum_open_interest_value',
    'sum_toptrader_long_short_ratio': 'sum_open_interest_value',
    'count_long_short_ratio': 'sum_open_interest_value',
    'sum_taker_long_short_vol_ratio': 'sum_open_interest_value'
}

base_agg_rules = {
    'sum_open_interest': 'last',
    'sum_open_interest_value': 'last',
}

def load_raw_data(category, source, start_date, end_date):
    file_path = os.path.join(bn_raw_path, f"{source}_data", BN_FUTURES_MAP[category],
                             category, 'metrics')
    file_path = Path(file_path)
    all_dfs = []
    for csv_file in file_path.glob('*/*.csv'):
        try:
            name = csv_file.name.split(".csv")[0]
            code = csv_file.parent.name
            if not (start_date <= name <= end_date):
                continue
            df = pd.read_csv(csv_file)
            df['code'] = code
            all_dfs.append(df)
            print(f"已加载: {csv_file.parent.name}/{csv_file.name}")
        except Exception as e:
            print(f"读取文件 {csv_file} 失败: {e}")

    if all_dfs:
        final_data = pd.concat(all_dfs, ignore_index=True)
        # 兼容不同列名：优先使用已有的 trade_time，或根据 create_time 转换为北京时间（UTC+8）
        time_col = 'trade_time' if 'trade_time' in final_data.columns else 'create_time'
        if time_col in final_data.columns:
            # 判断是数字时间戳还是时间字符串
            sample_val = final_data[time_col].dropna().iloc[0]
            if isinstance(sample_val, (int, float, np.number)) or str(sample_val).isdigit():
                val_len = len(str(int(sample_val)))
                unit = 'us' if val_len >= 16 else ('ms' if val_len >= 13 else 's')
                final_data['trade_time'] = (
                    pd.to_datetime(final_data[time_col], unit=unit, utc=True)
                    .dt.tz_convert('Asia/Shanghai')
                    .dt.tz_localize(None)
                )
            else:
                final_data['trade_time'] = (
                    pd.to_datetime(final_data[time_col], errors='coerce', utc=True)
                    .dt.tz_convert('Asia/Shanghai')
                    .dt.tz_localize(None)
                )
        final_data = final_data.sort_values(['code', 'trade_time'])
    return final_data if len(all_dfs) > 0 else pd.DataFrame()


def start(method, category, task_id):
    start_date, end_date = get_dates(method)
    final_data = load_raw_data(category=category,
                               source=TASK_MAPPING[task_id]['source'],
                               start_date=start_date,
                               end_date=end_date)
    if final_data.empty:
        print("未加载到 metrics 数据！")
        return

    # 构建当前任务专属的聚合规则字典
    current_agg_rules = base_agg_rules.copy()
    temp_prod_cols = []
    weight_cols = list(set(weighted_cols_map.values()))

    for target_col, weight_col in weighted_cols_map.items():
        if target_col in final_data.columns and weight_col in final_data.columns:
            prod_col_name = f"{target_col}_prod_weight"
            final_data[prod_col_name] = final_data[target_col] * final_data[weight_col]
            temp_prod_cols.append(prod_col_name)
            current_agg_rules[prod_col_name] = 'sum'

    for w_col in weight_cols:
        if w_col in final_data.columns:
            sum_col = f"{w_col}_sum_for_div"
            final_data[sum_col] = final_data[w_col]
            current_agg_rules[sum_col] = 'sum'

    # 按 code 和 1小时重采样
    resampled_data = final_data.groupby(
        ['code', pd.Grouper(key='trade_time', freq='1h')]
    ).agg(current_agg_rules).reset_index()

    # 还原加权值: 分子 / 分母
    for target_col, weight_col in weighted_cols_map.items():
        prod_col_name = f"{target_col}_prod_weight"
        weight_sum_name = f"{weight_col}_sum_for_div"
        if prod_col_name in resampled_data.columns and weight_sum_name in resampled_data.columns:
            resampled_data[target_col] = (
                resampled_data[prod_col_name] / resampled_data[weight_sum_name].replace(0, np.nan)
            )

    cols_to_drop = temp_prod_cols + [f"{w}_sum_for_div" for w in weight_cols]
    resampled_data.drop(columns=[c for c in cols_to_drop if c in resampled_data.columns], inplace=True)
    if 'create_time' in resampled_data.columns:
        resampled_data.drop(columns=['create_time'], inplace=True)
    pdb.set_trace()
    output_dirs = os.path.join(base_path, method, "basic", task_id)
    os.makedirs(output_dirs, exist_ok=True)
    filename = os.path.join(output_dirs, f"metrics_{category}.feather")
    print(filename)
    resampled_data.reset_index(drop=True).to_feather(filename)
    print(f"Metrics 数据保存成功，共 {len(resampled_data)} 行。")


if __name__ == '__main__':
    variant = Tactix().start()
    start(method=variant.method,
          category=variant.category,
          task_id=variant.task_id)