### 生成所用因子的绩效文件
import os, pdb
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


from kdutils.macro2 import *
from kdutils.tactix import Tactix

# SELECTED_MAPPING = {
#     '10001':  ["MMASSI(120,MNPOSITIVE(90,'corr_vwap_ask_price_0'),WMA(5,'smart_tick_in'))",
#                 "MVHF(10,MMASSI(120,MPRO(60,MVHF(10,MPRO(60,'money'))),MAPOSITIVE(10,'twap')))",
#                 "DELTA(120,MMAX(90,DELTA(90,'low')))",
#                 "RSI(120,MCPS(120,RSI(120,'pct_change_close')))",
#                 "MMAX(30,MQUANTILE(240,MMASSI(30,DELTA(90,MMIN(30,MIR(60,'smart_volume_out'))),MIChimoku(30,'mid_price_bias','delta_volume_bid1'))))",
#                 "DELTA(90,MMaxDiff(60,MADecay(120,MADecay(120,MADecay(120,'smart_tick_in_pct')))))",
#                 "MT3(90,MCPS(15,MSUM(120,'smart_tick_in')))",
#                 "MT3(30,MT3(90,MARGMIN(120,'twap')))",
#                 "MIR(10,MCORR(10,'pct_change_set','depth_imbalance_2'))",
#                 "MMASSI(90,'twap',MADecay(15,'order_flow_imbanlace_1'))",
#                 "MSharp(120,MRes(60,'smart_tick_out',MADiff(30,'delta_volume_ask1')),MCPS(120,'twap'))",
#                 "MINIMUM(MMedian(15,SIGLOG2ABS('pct_change_set')),SIGLOG2ABS('corr_ret_ask_price_0'))",
#                 "MCoef(5,MMaxDiff(5,'tick_out'),MKURT(10,'mid_price_bias_ratio'))",
#                 "MRANK(120, SUBBED(MRANK(30, DELTA(60, 'high')), MRANK(20, DELTA(5, 'high'))))",
#                 "DELTA(5,MT3(120,MCPS(120,MA(120,'pct_change'))))",
#                 "MA(120,MADiff(15,EMA(60,MSUM(60,'smart_money_in_pct'))))",
#                 "MIR(120,DELTA(90,MSKEW(90,DELTA(90,'high'))))",
#                 "MIR(120,MPERCENT(90,DELTA(90,'twap')))",
#                 "MHMA(30,MMedian(60,'smart_volume_in'))",
#                 "MRANK(30,MQUANTILE(15,MOD('pct_change','order_flow_imbanlace_avg5')))",
#                 "MMeanRes(60,MDEMA(30,'order_imbalance_ratio1'),MADecay(5,'order_imbalance_ratio1'))",
#                 "RSI(120,MCPS(120,MMedian(90,'pct_change_set')))",
#                 "MIR(15,DELTA(90,MA(90,DELTA(90,'twap'))))",
#                 "MMedian(90,MVHF(10,MT3(120,'pct_change')))",
#                 "DELTA(90,MMaxDiff(90,DELTA(90,SHIFT(60,'low'))))"]
# }


def parse_summary_file(file_path):
    KEY_MAPPING = {
    'Expression': 'expression',
    'Avg Return (bps)': 'avg_ret',
    'Total Return': 'total_ret',
    'Sharpe Ratio': 'sharpe',
    'Ann Sharpe Ratio': 'ann_sharpe',
    'Max Drawdown': 'max_dd',
    'Calmar Ratio': 'calmar',
    'Win Rate': 'win_rate',
    'Profit/Loss Ratio': 'pl_ratio',
    'IC Mean': 'ic_mean',
    'ICIR': 'icir',
    'Mean Turnover': 'turnover',
    'Factor Autocorr': 'factor_ac',
    'Return Autocorr': 'ret_ac'}

    data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line or line.startswith("---"): continue
            
            if line.startswith("Expression:"):
                data['Expression'] = line.split(":", 1)[1].strip()
                continue
                
            if ":" in line:
                key, value = line.split(":", 1)
                key, value = key.strip(), value.strip()
                try:
                    if value.endswith('%'):
                        clean_val = float(value.replace('%', ''))
                    else:
                        clean_val = float(value)
                    data[key] = clean_val
                except ValueError:
                    data[key] = value
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
    clean_data = {KEY_MAPPING.get(k, k): v for k, v in data.items()}
    return clean_data


def build(base_dir,  file_name,  dataset_type=None):
    records = []
    if dataset_type is None:
        target_types = ['train', 'test'] # 默认情况
    elif isinstance(dataset_type, str):
        target_types = [dataset_type]    # 如果传入字符串，转为列表
    else:
        target_types = dataset_type      # 假设传入的是列表
    
    print(f"正在加载以下类型的数据: {target_types} ...")
    pdb.set_trace()
    for current_root, dirs, files in os.walk(base_dir):
        if file_name in files:
            current_folder_name = os.path.basename(current_root)

            if current_folder_name not in target_types:
                continue

            file_path = os.path.join(current_root, file_name)
            parent_dir = os.path.dirname(current_root)
            experiment_id = os.path.basename(parent_dir)

            if file_name:
                file_data = parse_summary_file(file_path)

            file_data['factor_id'] = experiment_id
            file_data['dataset_type'] = current_folder_name
            # file_data['Full_Path'] = current_root # 调试用
            
            records.append(file_data)
    pdb.set_trace()
    return pd.DataFrame(records)


def run(method, instruments, task_id, period, param_id, 
        file_name, dataset_type, selected_id):
    path = os.path.join(base_path, method, instruments, 'temp', 'model', 
                str(task_id), str(period),'research','check',
                str(param_id))
    df_all = build(base_dir=path, file_name=file_name, dataset_type=dataset_type)
    # pdb.set_trace()
    # df_all = df_all[df_all.expression.isin(SELECTED_MAPPING['10001'])] if selected_id in SELECTED_MAPPING else df_all
    # print('-->')


if __name__ == '__main__':
    variant = Tactix().start()
    run(method=variant.method, instruments=variant.instruments,
        task_id=variant.task_id, period=variant.period,
        file_name=variant.filename,
        param_id=variant.param_id,
        dataset_type=variant.dataset_type,
        selected_id=variant.selected_id)