# BN 处理KLine数据
import pdb
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from lib.utils.macro import *
from lib.utils.tactix import Tactix
from lib.utils.ttimes import get_dates


def load_raw_data(category, source, period, start_date, end_date):
    file_path = os.path.join(
        bn_raw_path, f"{source}_data", BN_FUTURES_MAP[category], category, 'klines',
        period) if category != 'spot' else os.path.join(
            bn_raw_path, f"{source}_data", category, 'klines', period)
    pdb.set_trace()
    file_path = Path(file_path)
    pdb.set_trace()
    all_dfs = []
    # i = 0
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

            # i += 1
            # 将结果放入列表
            all_dfs.append(df)
            print(f"已加载: {csv_file.parent.name}/{csv_file.name}")
            # i += 1
            # if i > 1000:
            #    continue
        except Exception as e:
            print(f"读取文件 {csv_file} 失败: {e}")

    if all_dfs:
        final_data = pd.concat(all_dfs, ignore_index=True)
        # 币安源数据为 UTC 毫秒时间戳，转换为东八区（北京时间 UTC+8）与国内客户端一致
        unit = 'us' if category == 'spot' else 'ms'
        if 'trade_time' in final_data.columns:
            final_data['trade_time'] = (
                pd.to_datetime(final_data['open_time'], unit=unit, utc=True)
                .dt.tz_convert('Asia/Shanghai')
                .dt.tz_localize(None)  # 去掉时区信息，保留本地时间字符串格式，保证 feather 兼容
            )
        final_data = final_data.sort_values(['code', 'trade_time'])
    return final_data if len(all_dfs) > 0 else pd.DataFrame()


# 处理成标准的feather
def start(method, category, task_id):
    start_date, end_date = get_dates(method)
    final_data = load_raw_data(category=category,
                               source=TASK_MAPPING[task_id]['source'],
                               start_date=start_date,
                               end_date=end_date,
                               period=TASK_MAPPING[task_id]['period'])
    output_dirs = os.path.join(base_path, method, "basic", task_id)
    os.makedirs(output_dirs, exist_ok=True)
    filename = os.path.join(output_dirs, f"kline_{category}.feather")
    print(filename)
    drop_cols = [c for c in ['open_time', 'ignore'] if c in final_data.columns]
    final_data.drop(columns=drop_cols).reset_index(
        drop=True).to_feather(filename)


if __name__ == '__main__':
    variant = Tactix().start()
    start(method=variant.method,
          category=variant.category,
          task_id=variant.task_id)
