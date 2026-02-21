import pdb, os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from kdutils.ttimes import get_dates

weighted_cols_map = {
    'count_toptrader_long_short_ratio': 'sum_open_interest_value',
    'sum_toptrader_long_short_ratio': 'sum_open_interest_value',
    'count_long_short_ratio': 'sum_open_interest_value',
    'sum_taker_long_short_vol_ratio':
    'sum_open_interest_value'  # 暂时不要使用：没有五分钟K线如果有 quote_volume 请换成 quote_volume
}
agg_rules = {
    'code': 'first',
    'create_time': 'last',  # 修改为 last
    # 存量数据依然用 last
    'sum_open_interest': 'last',
    'sum_open_interest_value':
    'last',  # 注意：这个作为 1H 结果展示时用 last，但作为权重计算中间值时其实隐含了 sum
}


def load_raw_data(category, source, start_date, end_date):
    file_path = os.path.join(raw_path, f"{source}_data", FUTURES_MAP[category],
                             category, 'metrics')
    file_path = Path(file_path)
    all_dfs = []
    i = 0
    for csv_file in file_path.glob('*/*.csv'):
        try:
            # 读取 CSV 数据
            name = csv_file.name.split(".csv")[0]
            code = csv_file.parent.name
            if not (start_date <= name <= end_date):
                print(f"{code} {name} 不在时间范围内 {start_date}~{end_date}")
                continue
            df = pd.read_csv(csv_file)
            df['code'] = code

            # 将结果放入列表
            all_dfs.append(df)
            print(f"已加载: {csv_file.parent.name}/{csv_file.name}")
            #i += 1
            #print(i)
            #if i > 100:
            #    break
        except Exception as e:
            print(f"读取文件 {csv_file} 失败: {e}")

    if all_dfs:
        final_data = pd.concat(all_dfs, ignore_index=True)
        final_data['trade_time'] = pd.to_datetime(final_data['trade_time'],
                                                  errors='coerce',
                                                  format='ISO8601')
        final_data = final_data.sort_values(['code', 'trade_time'])
    return final_data if len(all_dfs) > 0 else pd.DataFrame()


def start(method, category, source, period):
    start_date, end_date = get_dates(method)
    final_data = load_raw_data(category=category,
                               source=source,
                               start_date=start_date,
                               end_date=end_date)

    temp_prod_cols = []
    weight_cols = list(set(weighted_cols_map.values()))  # 去重的权重列列表
    for target_col, weight_col in weighted_cols_map.items():
        # 生成中间列: Value * Weight
        prod_col_name = f"{target_col}_prod_weight"
        final_data[
            prod_col_name] = final_data[target_col] * final_data[weight_col]
        temp_prod_cols.append(prod_col_name)

    for col in temp_prod_cols:
        agg_rules[col] = 'sum'

    for w_col in weight_cols:
        final_data[f"{w_col}_sum_for_div"] = final_data[w_col]  # 复制一列专门用来求和
        agg_rules[f"{w_col}_sum_for_div"] = 'sum'

    resampled_data = final_data.groupby(
        ['symbol', pd.Grouper(key='trade_time',
                              freq='1h')]).agg(agg_rules).reset_index()
    for target_col, weight_col in weighted_cols_map.items():
        prod_col_name = f"{target_col}_prod_weight"
        weight_sum_name = f"{weight_col}_sum_for_div"

        # 分子 / 分母
        # 处理分母为0的情况，避免报错 (虽然持仓价值很难为0)
        resampled_data[target_col] = resampled_data[
            prod_col_name] / resampled_data[weight_sum_name].replace(
                0, np.nan)

    cols_to_drop = temp_prod_cols + [f"{w}_sum_for_div" for w in weight_cols]
    resampled_data.drop(columns=cols_to_drop, inplace=True)

    output_dirs = os.path.join(base_path, method, "basic", period, source)
    os.makedirs(output_dirs, exist_ok=True)
    filename = os.path.join(output_dirs, f"metrics_{category}.feather")
    resampled_data.to_feather(filename)


if __name__ == '__main__':
    variant = Tactix().start()
    start(
        method=variant.method,
        category='um',  #variant.category,
        source=variant.source,
        period=variant.period)
