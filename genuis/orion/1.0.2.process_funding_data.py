### BN处理funding数据
import pdb, os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import *
from kdutils.tactix import Tactix
from kdutils.ttimes import get_dates


def load_raw_data(category, source, start_date, end_date):
    file_path = os.path.join(raw_path, f"{source}_data", FUTURES_MAP[category],
                             category, 'fundingRate')
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

            i += 1
            # 将结果放入列表
            all_dfs.append(df)
            print(f"已加载: {csv_file.parent.name}/{csv_file.name}")
            #i += 1
            #if i > 100:
            #    continue
        except Exception as e:
            print(f"读取文件 {csv_file} 失败: {e}")

    if all_dfs:
        pdb.set_trace()
        final_data = pd.concat(all_dfs, ignore_index=True)
        final_data['trade_time'] = pd.to_datetime(final_data['trade_time'],
                                                  errors='coerce',
                                                  format='ISO8601')
        final_data = final_data.sort_values(['code', 'trade_time'])
    return final_data if len(all_dfs) > 0 else pd.DataFrame()


## 将资金费率平摊到每个小时
def start1(method, category, task_id):
    start_date, end_date = get_dates(method)
    final_data = load_raw_data(category=category,
                               source=TASK_MAPPING[task_id]['source'],
                               start_date=start_date,
                               end_date=end_date)
    pdb.set_trace()
    ## 将资金费率平摊到每个小时
    final_data['trade_time'] = pd.to_datetime(final_data['trade_time'],
                                              format='ISO8601').dt.floor('h')
    final_data['last_funding_rate'] = final_data[
        'last_funding_rate'] / final_data['funding_interval_hours']

    expanded_data = final_data.loc[final_data.index.repeat(
        final_data['funding_interval_hours'])].copy()
    offsets = expanded_data.groupby(level=0).cumcount()
    expanded_data[
        'trade_time'] = expanded_data['trade_time'] - pd.to_timedelta(offsets,
                                                                      unit='h')
    expanded_data = expanded_data.reset_index(drop=True)
    expanded_data = expanded_data.sort_values(['code', 'trade_time'])
    pdb.set_trace()
    output_dirs = os.path.join(base_path, method, "basic", task_id)
    os.makedirs(output_dirs, exist_ok=True)
    filename = os.path.join(output_dirs, f"funding_{category}.feather")
    expanded_data.drop(['calc_time'], axis=1).to_feather(filename)

def start2(method, category, task_id):
    start_date, end_date = get_dates(method)
    final_data = load_raw_data(category=category,
                               source=TASK_MAPPING[task_id]['source'],
                               start_date=start_date,
                               end_date=end_date)
    pdb.set_trace()
    # 1. 对齐时间到小时
    final_data['trade_time'] = pd.to_datetime(final_data['trade_time'], format='ISO8601').dt.floor('h')
    
    # 2. 清理空时间和去重 (极度重要，防止任何隐藏的合并错乱)
    final_data = final_data.dropna(subset=['trade_time'])
    final_data = final_data.drop_duplicates(subset=['trade_time', 'code'])
    
    # 3. 排序 (代替原本复杂的重组)
    final_data = final_data.sort_values(['code', 'trade_time'])
    
    # --- 只要这 3 步，数据就已经干净了，直接保存！ ---
    # 去掉没用的 calc_time
    if 'calc_time' in final_data.columns:
        final_data = final_data.drop(['calc_time'], axis=1)
        
    # 4. 保存为 feather
    output_dirs = os.path.join(base_path, method, "basic", task_id)
    os.makedirs(output_dirs, exist_ok=True)
    filename = os.path.join(output_dirs, f"funding_{category}.feather")
    
    final_data.reset_index(drop=True).to_feather(filename)
    print(f"资金费率保存成功，共 {len(final_data)} 行。")
    
    
    
    
if __name__ == '__main__':
    variant = Tactix().start()
    start2(method=variant.method,
          category=variant.category,
          task_id=variant.task_id)
