import os, pdb
from pathlib import Path
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

from kdutils.macro2 import base_path
from lib.flp001 import *


# def parse_summary_file(file_path):
#     KEY_MAPPING = {
#     'Name': 'name',
#     'Expression': 'expression',
#     'Avg Return (bps)': 'avg_ret',
#     'Total Return': 'total_ret',
#     'Sharpe Ratio': 'sharpe',
#     'Ann Sharpe Ratio': 'ann_sharpe',
#     'Max Drawdown': 'max_dd',
#     'Calmar Ratio': 'calmar',
#     'Win Rate': 'win_rate',
#     'Profit/Loss Ratio': 'pl_ratio',
#     'IC Mean': 'ic_mean',
#     'ICIR': 'icir',
#     'Mean Turnover': 'turnover',
#     'Factor Autocorr': 'factor_ac',
#     'Return Autocorr': 'ret_ac'
#     }

#     data = {}
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             lines = f.readlines()
#         for line in lines:
#             line = line.strip()
#             if not line or line.startswith("---"): continue
            
#             if line.startswith("Expression:"):
#                 data['Expression'] = line.split(":", 1)[1].strip()
#                 continue
                
#             if ":" in line:
#                 key, value = line.split(":", 1)
#                 key, value = key.strip(), value.strip()
#                 try:
#                     if value.endswith('%'):
#                         clean_val = float(value.replace('%', ''))
#                     else:
#                         clean_val = float(value)
#                     data[key] = clean_val
#                 except ValueError:
#                     data[key] = value
#     except Exception as e:
#         print(f"Error parsing file {file_path}: {e}")
#     clean_data = {KEY_MAPPING.get(k, k): v for k, v in data.items()}
#     return clean_data


def test1():
    method = 'bicso2'
    task_id = '113001'
    period = 5
    session = "20260325"
    instruments = 'rbb'
    pdb.set_trace()
    file_path = os.path.join(base_path, method, instruments, 'rulex', task_id,
                             "nxt1_ret_{0}h".format(period),
                             "d{0}".format(session))
    file_path = Path(file_path)
    res = []
    for feather_file in file_path.rglob('*.txt'):
        data1 = parse_summary_file(feather_file)
        res.append(data1)
    results = pd.DataFrame(res)
    results['name'] = results['name'].astype(int).astype(str)
    pdb.set_trace()
    print('-->')

def test2():
    method = 'bicso2'
    task_id = '113001'
    period = 5
    instruments = 'rbb'
    sessions = ['20260325', '20260401']
    base_dirs = os.path.join(base_path, method, instruments, 'rulex', task_id,
                             "nxt1_ret_{0}h".format(period))

    drafit_data = pd.read_csv(os.path.join(base_dirs, "draft.csv"))
    res1 = []
    for session in sessions:
        file_path = Path(os.path.join(base_dirs, "d{0}".format(session)))
        res = []
        for feather_file in file_path.rglob('*.txt'):
            data1 = parse_summary_file1(feather_file)
            pdb.set_trace()
            res.append(data1)
        results = pd.DataFrame(res)
        results['name'] = results['name'].astype(int).astype(str)
        res1.append(results)
    
    results1 = pd.concat(res1, axis=0)
    results1 = results1[results1['expression'].isin(drafit_data['formula'].to_list())]
    pdb.set_trace()
    print('--->')

def test3():
    method = 'cicso0'
    task_id = '200037'
    period = 15
    instruments = 'ims'
    base_dirs = os.path.join(base_path, method, instruments, 'rulex', task_id,
                             "nxt1_ret_{0}h".format(period))

    file_path = Path(base_dirs)
    drafit_data = pd.read_csv(os.path.join(base_dirs, "draft.csv"))
    file_path = Path(file_path)
    res = []
    for feather_file in file_path.rglob('*.txt'):
        data1 = parse_summary(feather_file)
        res.append(data1)
    # for session in sessions:
    #     file_path = Path(os.path.join(base_dirs, "d{0}".format(session)))
    #     res = []
    #     for feather_file in file_path.rglob('*.txt'):
    #         data1 = parse_summary_file(feather_file)
    #         res.append(data1)
    #     results = pd.DataFrame(res)
    #     results['name'] = results['name'].astype(int).astype(str)
    #     res1.append(results)
    pdb.set_trace()
    results1 = pd.DataFrame(res)
    results1 = results1.drop(['roll_win','resampling_win','holding_profit'],axis=1)
    results1 = results1[results1['expression'].isin(drafit_data['formula'].to_list())]
    pdb.set_trace()
    print('--->')



test3()
