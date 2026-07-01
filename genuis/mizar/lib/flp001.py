import os, re, pdb
from IPython.display import display, HTML
from pathlib import Path
import pandas as pd
from kdutils.macro2 import base_path
from lib.iux002 import create_id, generate_simple_id

STANDARD_SCHEMA = {
    'name': None,
    'expression': None,
    'avg_ret': None,
    'total_ret': None,
    'sharpe': None,
    'ann_sharpe': None,
    'max_dd': None,
    'calmar': None,
    'win_rate': None,
    'pl_ratio': None,
    'ic_mean': None,
    'icir': None,
    'turnover': None,
    'factor_ac': None,
    'ret_ac': None,
    'roll_win': None,
    'resampling_win': None,
    'holding_profit': None
}

def parse_summary(file_path):
    """
    智能解析入口：根据路径自动选择 file1 或 file2 解析器
    """
    # 确保传入的是 Path 对象 (兼容传入字符串的情况)
    if isinstance(file_path, str):
        path_obj = Path(file_path)
    else:
        path_obj = file_path

    # 判断逻辑：遍历路径的所有层级 (parts)，检查是否有以 'd' 开头且后面跟着数字的目录 (如 d202523122)
    # path_obj.parts 会把路径拆成元组: ('records', 'cicso0', ..., 'd202523122', '10908278', 'performance_summary.txt')
    has_d_dir = any(re.match(r'^d\d+', part) for part in path_obj.parts)

    # 按照你的需求：有 d 使用 parse_summary_file1，没有 d 使用 parse_summary_file2
    if has_d_dir:
        # print(f"检测到 d 目录，使用解析器 1 -> {path_obj}")
        dt1 =  parse_summary_file1(file_path)
        dt1['category'] = 'd'
    else:
        # print(f"未检测到 d 目录，使用解析器 2 -> {path_obj}")
        dt1 = parse_summary_file2(file_path)
        dt1['category'] = 'p'
    return dt1


def parse_summary_file1(file_path):
    """
    处理老格式 (带有 Name: xxx 这种)
    """
    KEY_MAPPING = {
        'Name': 'name',
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
        'Return Autocorr': 'ret_ac'
    }

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
        
    # --- 【关键修改点】：使用标准模板进行填充 ---
    # 先做映射
    mapped_data = {KEY_MAPPING.get(k, k): v for k, v in data.items()}
    # 拷贝一份全集模板
    final_data = STANDARD_SCHEMA.copy()
    # 把解析到的数据合并进去 (多余的脏 key 比如 'Factor Comparison' 会被忽略)
    for k in final_data.keys():
        if k in mapped_data:
            final_data[k] = mapped_data[k]
            
    return final_data


def parse_summary_file2(file_path):
    """
    处理新格式对比表格 (ims vs. ics)
    """
    KEY_MAPPING = {
        'Name': 'name',
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
        'Return Autocorr': 'ret_ac',
        'Roll Window': 'roll_win',
        'Resampling Window': 'resampling_win',
        'Holding Profit': 'holding_profit',
    }

    data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith("---") or line.startswith("Metric"): 
                continue
            
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    key = parts[0].strip()
                    value = parts[1].strip()  
                    
            elif ":" in line:
                parts = line.split(":", 1)
                key = parts[0].strip()
                value = parts[1].strip()
                
            else:
                continue 

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
        
    # --- 【关键修改点】：使用标准模板进行填充 ---
    mapped_data = {KEY_MAPPING.get(k, k): v for k, v in data.items()}
    final_data = STANDARD_SCHEMA.copy()
    
    # 额外逻辑：老格式有 Name 属性记录的是目录数字，新格式文件可能没有 Name 这行。
    # 我们可以从 file_path 也就是倒数第二级目录提取 Name 来保证一致性。
    import os
    parent_dir_name = os.path.basename(os.path.dirname(file_path))
    if 'Name' not in data and parent_dir_name.isdigit():
        mapped_data['name'] = float(parent_dir_name) # 为了跟你图里的格式保持 float
        
    for k in final_data.keys():
        if k in mapped_data:
            final_data[k] = mapped_data[k]
            
    return final_data


def load_data(method, task_id, instruments, period, session, category):
    session_name = "d{0}".format(session) if category ==2 else "{0}".format(session) 
    file_path = os.path.join(base_path, method, instruments, 'rulex', task_id,
                             "nxt1_ret_{0}h".format(period),
                             "{0}".format(session_name))
    print(file_path)
    file_path = Path(file_path)
    res = []
    for feather_file in file_path.rglob('*.txt'):
        data1 = parse_summary(feather_file)
        res.append(data1)
    results = pd.DataFrame(res)
    return results

def load_data2(method, task_id, instruments, period, filename="draft.csv"):
    res = []
    file_path = os.path.join(base_path, method, instruments, 'rulex', task_id,
                             "nxt1_ret_{0}h".format(period))
    draft_data = pd.read_csv(os.path.join(file_path, filename))
    draft_data['source'] = draft_data['source'].astype(int)
    draft_data['factor_id'] = draft_data['formula'].apply(lambda x: create_id(generate_simple_id(x)))
    for row in draft_data.itertuples():
        filename = os.path.join(file_path, "d{}".format(
            row.source) if row.category=='d' else "{}".format(row.source),
                                row.factor_id, "performance_summary.txt")
        data1 = parse_summary(filename)
        desired_order = [
            'factor_id', 'formula', 'category', 'direction', 'source', 
            'ic_mean', 'ann_sharpe', 'calmar', 'max_dd', 'avg_ret', 
            'total_ret', 'win_rate', 'pl_ratio', 'turnover', 'factor_ac','plot']
        
        target_keys = ['avg_ret', 'ann_sharpe', 'max_dd', 'calmar', 
                       'win_rate','pl_ratio','ic_mean', 'turnover',
                       'factor_ac','total_ret']
        extracted_data = {k: data1.get(k) for k in target_keys}
        extracted_data.update(row._asdict())
        extracted_data["plot"] = os.path.join(file_path, "d{}".format(
            row.source) if row.category=='d' else "{}".format(row.source),
                                row.factor_id, "evaluation_plot.png")
        ordered_data = {k: extracted_data.get(k) for k in desired_order}
        res.append(ordered_data)
    return pd.DataFrame(res)
    
    
def load_data3(method, task_id, instruments, period, filename="draft.csv"):
    res = []
    file_path = os.path.join(base_path, method, instruments, 'rulex', task_id,
                             "nxt1_ret_{0}h".format(period))
    draft_data = pd.read_csv(os.path.join(file_path, filename))
    draft_data['source'] = draft_data['source'].astype(int)
    draft_data['factor_id'] = draft_data['formula'].apply(lambda x: create_id(generate_simple_id(x)))
    for row in draft_data.itertuples():
        filename = os.path.join(file_path, "recent",
                                row.factor_id, "performance_summary.txt")
        data1 = parse_summary(filename)
        desired_order = [
            'factor_id', 'formula', 'category', 'direction', 'source', 
            'ic_mean', 'ann_sharpe', 'calmar', 'max_dd', 'avg_ret', 
            'total_ret', 'win_rate', 'pl_ratio', 'turnover', 'factor_ac','plot']
        
        target_keys = ['avg_ret', 'ann_sharpe', 'max_dd', 'calmar', 
                       'win_rate','pl_ratio','ic_mean', 'turnover',
                       'factor_ac','total_ret']
        extracted_data = {k: data1.get(k) for k in target_keys}
        extracted_data.update(row._asdict())
        extracted_data["plot"] = os.path.join(file_path, "recent",
                                row.factor_id, "evaluation_plot.png")
        ordered_data = {k: extracted_data.get(k) for k in desired_order}
        res.append(ordered_data)
    return pd.DataFrame(res)

def make_clickable(val):
    return f'<a target="_blank" href="{val}">{val}</a>'


def to_html(results):
    return display(HTML(results.to_html(escape=False)))