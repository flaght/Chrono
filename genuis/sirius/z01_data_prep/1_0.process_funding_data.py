### BN处理funding数据
import pdb, os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from lib.utils.macro import *
from lib.utils.tactix import Tactix
from lib.utils.ttimes import get_dates

# from kdutils.macro2 import *
# from kdutils.tactix import Tactix
# from kdutils.ttimes import get_dates


def load_raw_data(category, source, start_date, end_date):
    pdb.set_trace()
    file_path = os.path.join(bn_raw_path, f"{source}_data", BN_FUTURES_MAP[category],
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
        final_data = pd.concat(all_dfs, ignore_index=True)
        # 兼容不同数据格式：优先处理已有 trade_time，或根据 calc_time/funding_time 位数自适应单位转换为北京时间（UTC+8）
        if 'trade_time' in final_data.columns and not final_data['trade_time'].isna().all():
            final_data['trade_time'] = (
                pd.to_datetime(final_data['trade_time'], errors='coerce', utc=True)
                .dt.tz_convert('Asia/Shanghai')
                .dt.tz_localize(None)
            )
        elif 'calc_time' in final_data.columns:
            val = final_data['calc_time'].dropna().iloc[0]
            val_len = len(str(int(val)))
            unit = 'us' if val_len >= 16 else ('ms' if val_len >= 13 else 's')
            final_data['trade_time'] = (
                pd.to_datetime(final_data['calc_time'], unit=unit, utc=True)
                .dt.tz_convert('Asia/Shanghai')
                .dt.tz_localize(None)
            )
        final_data = final_data.sort_values(['code', 'trade_time'])
    return final_data if len(all_dfs) > 0 else pd.DataFrame()


# 处理资金费率成标准的 feather (保留真实结算点，不平摊)
def start(method, category, task_id):
    start_date, end_date = get_dates(method)
    final_data = load_raw_data(category=category,
                               source=TASK_MAPPING[task_id]['source'],
                               start_date=start_date,
                               end_date=end_date)
    if final_data.empty:
        print("未加载到资金费率数据！")
        return

    # 1. 对齐时间到小时整点
    final_data['trade_time'] = final_data['trade_time'].dt.floor('h')
    pdb.set_trace()
    # 2. 清理空时间和去重
    final_data = final_data.dropna(subset=['trade_time'])
    final_data = final_data.drop_duplicates(subset=['trade_time', 'code'])

    # 3. 标准化列名：下游统一使用 f_funding_rate
    if 'last_funding_rate' in final_data.columns:
        final_data = final_data.rename(columns={'last_funding_rate': 'f_funding_rate'})

    # 4. 排序
    final_data = final_data.sort_values(['code', 'trade_time'])

    # 5. 清理多余列
    drop_cols = [c for c in ['calc_time'] if c in final_data.columns]
    if drop_cols:
        final_data = final_data.drop(columns=drop_cols)

    # 6. 保存为 feather
    output_dirs = os.path.join(base_path, method, "basic", task_id)
    os.makedirs(output_dirs, exist_ok=True)
    filename = os.path.join(output_dirs, f"funding_{category}.feather")
    print(filename)
    final_data.reset_index(drop=True).to_feather(filename)
    print(f"资金费率保存成功，共 {len(final_data)} 行。")


if __name__ == '__main__':
    variant = Tactix().start()
    start(method=variant.method,
          category=variant.category,
          task_id=variant.task_id)
