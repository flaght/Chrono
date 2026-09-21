import os, re, pdb
from html import escape, unescape
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, HTML
from concurrent.futures import ThreadPoolExecutor
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
        dt1 = parse_summary_file1(file_path)
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
        mapped_data['name'] = float(parent_dir_name)  # 为了跟你图里的格式保持 float

    for k in final_data.keys():
        if k in mapped_data:
            final_data[k] = mapped_data[k]

    return final_data


def load_signal_performance(base_dir, data_type=None):
    """加载信号回测落盘绩效，供 Notebook 独立查询。

    Parameters
    ----------
    base_dir : str or pathlib.Path
        可以传入 ``.../rl``、``.../rl/composite``，也可以直接传入
        composite 下某个模型目录。函数会递归查找 performance_summary.txt。
    data_type : {None, 'optimi', 'obse', 'obs'}
        ``optimi`` 是参数寻优集；``obse`` 是参数验证集，，``obs`` 是
        ``obse`` 的查询别名。None 表示同时加载两类数据。

    Returns
    -------
    pandas.DataFrame
        每个模型、品种、信号参数和数据类型一行，同时包含原始绩效文件
        与 evaluation_plot.png 的绝对路径。
    """
    aliases = {
        "optimi": "optimi",
        "obse": "obse",
        "obs": "obse",
        "test": "trst"
    }
    selected_type = None
    if data_type is not None:
        selected_type = aliases.get(str(data_type).lower())
        if selected_type is None:
            raise ValueError("data_type只允许None、optimi、obse或obs")

    search_root = Path(base_dir).expanduser()
    if not search_root.exists():
        raise FileNotFoundError(f"绩效根目录不存在: {search_root}")
    if search_root.name != "composite" and (search_root /
                                            "composite").is_dir():
        search_root = search_root / "composite"

    records = []
    for summary_path in sorted(search_root.rglob("performance_summary.txt")):
        result_name = summary_path.parent.name
        segment = next((canonical for suffix, canonical in aliases.items()
                        if result_name.endswith(f"_{suffix}")), None)
        if segment is None or (selected_type is not None
                               and segment != selected_type):
            continue

        # 标准目录：model/signal_method/signal_id/result_name/summary。
        # 从文件向上解析，因此base_dir既可以是rl/composite，也可以直接
        # 指向某个模型目录。
        result_dir = summary_path.parent
        signal_id_dir = result_dir.parent
        signal_method_dir = signal_id_dir.parent
        model_dir = signal_method_dir.parent
        if not all((result_dir.name, signal_id_dir.name,
                    signal_method_dir.name, model_dir.name)):
            continue
        model = model_dir.name
        signal_method = signal_method_dir.name
        signal_id = signal_id_dir.name
        parsed = parse_summary(summary_path)
        plot_path = summary_path.with_name("evaluation_plot.png")

        def percent_to_ratio(value):
            return None if value is None else float(value) / 100.0

        records.append({
            "signal_method":
            signal_method,
            "signal_id":
            str(signal_id),
            "model":
            model,
            "code":
            result_name.split("_", 1)[0],
            "segment":
            segment,
            "data_type": ("参数寻优集" if segment == "optimi" else
                          ("固定参数集" if segment == 'obse' else "测试集")),
            "avg_ret":
            parsed.get("avg_ret"),
            "total_ret":
            percent_to_ratio(parsed.get("total_ret")),
            "sharpe":
            parsed.get("sharpe"),
            "ann_sharpe":
            parsed.get("ann_sharpe"),
            "max_dd":
            percent_to_ratio(parsed.get("max_dd")),
            "calmar":
            parsed.get("calmar"),
            "win_rate":
            percent_to_ratio(parsed.get("win_rate")),
            "pl_ratio":
            parsed.get("pl_ratio"),
            "ic_mean":
            parsed.get("ic_mean"),
            "icir":
            parsed.get("icir"),
            "turnover":
            parsed.get("turnover"),
            "factor_ac":
            parsed.get("factor_ac"),
            "ret_ac":
            parsed.get("ret_ac"),
            "performance_file":
            str(summary_path.resolve()),
            "plot":
            str(plot_path.resolve()) if plot_path.exists() else None,
        })

    if not records:
        suffix = "全部数据类型" if selected_type is None else selected_type
        raise FileNotFoundError(f"在{search_root}下没有找到{suffix}的信号绩效文件")
    return (pd.DataFrame(records).sort_values(
        ["segment", "signal_method", "signal_id", "model",
         "code"]).reset_index(drop=True))


def parse_comparison_summary(file_path):
    """解析双品种对比表，每个品种返回一条记录，百分数保留百分数值。"""
    key_mapping = {
        'Avg Return (bps)': 'avg_ret',
        'Total Return': 'total_ret',
        'Sharpe Ratio': 'sharpe',
        'Ann Sharpe Ratio': 'ann_sharpe',
        'Max Drawdown': 'max_dd',
        'Calmar Ratio': 'calmar',
        'Win Rate': 'win_rate',
        'Profit/Loss Ratio': 'pl_ratio',
        'IC Mean': 'ic_mean',
        'Total IC': 'total_ic',
        'ICIR': 'icir',
        'Mean Turnover': 'turnover',
        'Factor Autocorr': 'factor_ac',
        'Return Autocorr': 'ret_ac',
        'Roll Window': 'roll_win',
        'Resampling Window': 'resampling_win',
        'Holding Profit': 'holding_profit',
        'Expression': 'expression',
        'Name': 'name',
    }

    def parse_value(value):
        try:
            return float(value.rstrip('%'))
        except ValueError:
            return value or None

    path = Path(file_path)
    shared = STANDARD_SCHEMA.copy()
    shared['total_ic'] = None
    shared['category'] = ('d' if any(
        re.match(r'^d\d+', part) for part in path.parts) else 'p')
    if path.parent.name.isdigit():
        shared['name'] = float(path.parent.name)
    instruments = []
    metrics = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith('---'):
                continue
            # 先处理元数据，避免表达式中的 | 被误识别为表格分隔符。
            if ':' in line:
                label, value = (part.strip() for part in line.split(':', 1))
                key = key_mapping.get(label)
                if key:
                    shared[key] = (value if key in ('expression',
                                                    'holding_profit') else
                                   parse_value(value))
            elif '|' in line:
                parts = [part.strip() for part in line.split('|')]
                if parts[0] == 'Metric':
                    instruments = parts[1:]
                elif parts[0] in key_mapping:
                    metrics[key_mapping[parts[0]]] = parts[1:]

    if len(instruments) != 2 or not all(instruments):
        raise ValueError(f'{file_path}: 需要包含两个品种的 Metric 表头')
    records = [
        dict(shared, instrument=instrument) for instrument in instruments
    ]
    for key, values in metrics.items():
        if len(values) != 2:
            raise ValueError(f'{file_path}: 指标 {key} 需要两个品种的值')
        for record, value in zip(records, values):
            record[key] = parse_value(value)
    return records


def load_data(method, task_id, instruments, period, session, category):
    session_name = "d{0}".format(session) if category == 2 else "{0}".format(
        session)
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
    draft_data['factor_id'] = draft_data['formula'].apply(
        lambda x: create_id(generate_simple_id(x)))
    for row in draft_data.itertuples():
        filename = os.path.join(
            file_path, "d{}".format(row.source) if row.category == 'd' else
            "{}".format(row.source), row.factor_id, "performance_summary.txt")
        data1 = parse_summary(filename)
        desired_order = [
            'factor_id', 'formula', 'category', 'direction', 'source',
            'ic_mean', 'ann_sharpe', 'calmar', 'max_dd', 'avg_ret',
            'total_ret', 'win_rate', 'pl_ratio', 'turnover', 'factor_ac',
            'plot'
        ]

        target_keys = [
            'avg_ret', 'ann_sharpe', 'max_dd', 'calmar', 'win_rate',
            'pl_ratio', 'ic_mean', 'turnover', 'factor_ac', 'total_ret'
        ]
        extracted_data = {k: data1.get(k) for k in target_keys}
        extracted_data.update(row._asdict())
        name = "comparison_plot.png" if extracted_data[
            'category'] == 'p' else "evaluation_plot.png"
        extracted_data["plot"] = os.path.join(
            file_path, "d{}".format(row.source) if row.category == 'd' else
            "{}".format(row.source), row.factor_id, name)
        ordered_data = {k: extracted_data.get(k) for k in desired_order}
        res.append(ordered_data)
    return pd.DataFrame(res)


def load_data3(method,
               task_id,
               instruments,
               period,
               category=None,
               filename="draft.csv"):
    res = []
    file_path = os.path.join(base_path, method, instruments, 'rulex', task_id,
                             "nxt1_ret_{0}h".format(period))
    draft_data = pd.read_csv(os.path.join(file_path, filename))
    draft_data['source'] = draft_data['source'].astype(int)
    draft_data['factor_id'] = draft_data['formula'].apply(
        lambda x: create_id(generate_simple_id(x)))
    if isinstance(category, str):
        draft_data = draft_data[draft_data['category'] == category]
    #print(draft_data)
    for row in draft_data.itertuples():
        filename = os.path.join(file_path, "recent", row.factor_id,
                                "performance_summary.txt")
        data1 = parse_summary(filename)
        desired_order = [
            'factor_id', 'formula', 'category', 'direction', 'source',
            'ic_mean', 'ann_sharpe', 'calmar', 'max_dd', 'avg_ret',
            'total_ret', 'win_rate', 'pl_ratio', 'turnover', 'factor_ac',
            'plot'
        ]

        target_keys = [
            'avg_ret', 'ann_sharpe', 'max_dd', 'calmar', 'win_rate',
            'pl_ratio', 'ic_mean', 'turnover', 'factor_ac', 'total_ret'
        ]
        extracted_data = {k: data1.get(k) for k in target_keys}
        extracted_data.update(row._asdict())

        name = "comparison_plot.png" if extracted_data[
            'category'] == 'p' else "evaluation_plot.png"
        extracted_data["plot"] = os.path.join(file_path, "recent",
                                              row.factor_id, name)
        ordered_data = {k: extracted_data.get(k) for k in desired_order}
        res.append(ordered_data)
    return pd.DataFrame(res)


def load_data4(method,
               task_id,
               instruments,
               period,
               category=None,
               filename="draft.csv",
               splits=('train', 'val', 'recent')):
    res = []
    file_path = os.path.join(base_path, method, instruments, 'rulex', task_id,
                             f"nxt1_ret_{period}h")
    draft_data = pd.read_csv(os.path.join(file_path, filename))
    draft_data['source'] = draft_data['source'].astype(int)
    draft_data['factor_id'] = draft_data['formula'].apply(
        lambda x: create_id(generate_simple_id(x)))
    if isinstance(category, str):
        draft_data = draft_data[draft_data['category'] == category]
    desired_order = [
        'factor_id', 'formula', 'category', 'direction', 'source', 'split',
        'ic_mean', 'ann_sharpe', 'calmar', 'max_dd', 'avg_ret', 'total_ret',
        'win_rate', 'pl_ratio', 'turnover', 'factor_ac', 'plot'
    ]
    target_keys = [
        'avg_ret', 'ann_sharpe', 'max_dd', 'calmar', 'win_rate', 'pl_ratio',
        'ic_mean', 'turnover', 'factor_ac', 'total_ret'
    ]
    for row in draft_data.itertuples():
        # 确定图片文件名
        plot_name = "comparison_plot.png" if row.category == 'p' else "evaluation_plot.png"
        # 遍历 train, val, recent，每个 split 独立生成一行
        for split in splits:
            split_row = {
                'factor_id': row.factor_id,
                'formula': row.formula,
                'category': row.category,
                'direction': getattr(row, 'direction', None),
                'source': row.source,
                'split': split,  # 标注是 train / val / recent
            }
            # 自动兼容是否有 'splits' 子目录层级
            split_dir = os.path.join(file_path, "splits", split, row.factor_id)
            if not os.path.exists(split_dir):
                split_dir = os.path.join(file_path, split, row.factor_id)
            # 1. 读取绩效 txt
            summary_file = os.path.join(split_dir, "performance_summary.txt")
            if os.path.exists(summary_file):
                summary_data = parse_summary(summary_file)
                for k in target_keys:
                    split_row[k] = summary_data.get(k)
            else:
                for k in target_keys:
                    split_row[k] = None
            # 2. 记录图片路径
            img_path = os.path.join(split_dir, plot_name)
            split_row['plot'] = img_path if os.path.exists(img_path) else None
            # 按统一字段顺序保存单行
            ordered_data = {k: split_row.get(k) for k in desired_order}
            res.append(ordered_data)
    return pd.DataFrame(res)


## 整个目录检索文件
def fetch_data(method, task_id, instruments, period, session, category):
    """按 load_data 的目录规则加载双品种绩效，每个文件对应两行。

    instrument 标识品种，共享 name、expression 和窗口信息；
    total_ret、max_dd 等百分比字段与 load_data 一致，不除以 100。
    """
    session_name = f'd{session}' if category == 2 else str(session)
    file_path = Path(base_path) / method / instruments / 'rulex' / task_id / (
        f'nxt1_ret_{period}h') / session_name
    print(file_path)
    records = []
    for summary_file in sorted(file_path.rglob('*.txt')):
        records.extend(parse_comparison_summary(summary_file))
    return pd.DataFrame(
        records,
        columns=[*STANDARD_SCHEMA, 'total_ic', 'category', 'instrument'])


def fetch_data2(method, task_id, instruments, period, filename="cohort.csv"):
    file_path = os.path.join(base_path, method, instruments, 'rulex', task_id,
                             "nxt1_ret_{0}h".format(period))
    cohort_data = pd.read_csv(os.path.join(file_path, filename))
    cohort_data['source'] = cohort_data['source'].astype(int)
    cohort_data['factor_id'] = cohort_data['formula'].apply(
        lambda x: create_id(generate_simple_id(x)))

    records = []
    for row in cohort_data.itertuples():
        summary_file = os.path.join(
            file_path, "d{}".format(row.source)
            if row.category == 'd' else "{}".format(row.source),
            str(row.factor_id), "performance_summary.txt")
        if os.path.exists(summary_file):
            recs = parse_comparison_summary(summary_file)
            for rec in recs:
                if not rec.get('expression') and hasattr(row, 'formula'):
                    rec['expression'] = row.formula
                if not rec.get('name') and hasattr(row, 'factor_id'):
                    rec['name'] = row.factor_id
            records.extend(recs)
        else:
            print(f"Warning: {summary_file} not found")

    df = pd.DataFrame(
        records,
        columns=[*STANDARD_SCHEMA, 'total_ic', 'category', 'instrument'])

    return df


def fetch_data3(method,
                task_id,
                instruments,
                period,
                category='recent',
                filename="cohort.csv"):
    file_path = os.path.join(base_path, method, instruments, 'rulex', task_id,
                             "nxt1_ret_{0}h".format(period))
    cohort_data = pd.read_csv(os.path.join(file_path, filename))
    cohort_data['source'] = cohort_data['source'].astype(int)
    cohort_data['factor_id'] = cohort_data['formula'].apply(
        lambda x: create_id(generate_simple_id(x)))

    records = []
    for row in cohort_data.itertuples():
        summary_file = os.path.join(file_path, category, row.factor_id,
                                    "performance_summary.txt")
        if os.path.exists(summary_file):
            recs = parse_comparison_summary(summary_file)
            for rec in recs:
                if not rec.get('expression') and hasattr(row, 'formula'):
                    rec['expression'] = row.formula
                if not rec.get('name') and hasattr(row, 'factor_id'):
                    rec['name'] = row.factor_id
            records.extend(recs)
        else:
            print(f"Warning: {summary_file} not found")

    df = pd.DataFrame(
        records,
        columns=[*STANDARD_SCHEMA, 'total_ic', 'category', 'instrument'])

    return df


def fetch_data4(method,
                task_id,
                instruments,
                period,
                windows=None,
                filename=None,
                category=None):

    file_path = os.path.join(base_path, method, instruments, 'rulex', task_id,
                             "nxt1_ret_{0}h".format(period))
    cohort_data = pd.read_csv(os.path.join(file_path, filename))
    cohort_data['source'] = cohort_data['source'].astype(int)
    cohort_data['factor_id'] = cohort_data['formula'].apply(
        lambda x: create_id(generate_simple_id(x)))
    records = []
    for row in cohort_data.itertuples():
        summary_file = os.path.join(file_path, category, windows,
                                    row.factor_id, "performance_summary.txt")
        if os.path.exists(summary_file):
            recs = parse_comparison_summary(summary_file)
            for rec in recs:
                if not rec.get('expression') and hasattr(row, 'formula'):
                    rec['expression'] = row.formula
                if not rec.get('name') and hasattr(row, 'factor_id'):
                    rec['name'] = row.factor_id
            records.extend(recs)
        else:
            print(f"Warning: {summary_file} not found")

    df = pd.DataFrame(
        records,
        columns=[*STANDARD_SCHEMA, 'total_ic', 'category', 'instrument'])

    return df


def make_clickable(val):
    return f'<a target="_blank" href="{val}">{val}</a>'


def extract_href(value):
    """兼容纯路径和已经生成的 HTML 超链接。"""
    text = str(value)

    match = re.search(r'''href=["']([^"']+)["']''', text)
    if match:
        return unescape(match.group(1))

    return unescape(text)


def to_html1(results, u_name, k_name=[]):
    data = results.copy()
    temp_id = data
    data['id'] = [
        '_'.join(map(str, row)) for row in data[k_name].itertuples(index=False)
    ]
    data['id'] = data['id'].apply(lambda x: create_id(generate_simple_id(x)))
    data[u_name] = data.apply(
        lambda row:
        (f'<a href="{escape(extract_href(row["plot"]), quote=True)}">'
         f'{escape(str(row["id"]))}</a>'),
        axis=1,
    )
    return display(HTML(data.to_html(escape=False, index=False)))


def to_html(results):
    data = results.copy()

    data["url"] = data.apply(
        lambda row:
        (f'<a href="{escape(extract_href(row["plot"]), quote=True)}">'
         f'{escape(str(row["factor_id"]))}</a>'),
        axis=1,
    )
    if 'source' in data.columns and 'direction' in data.columns:
        data = data[[
            "url", "direction", "formula", "source", "plot", "factor_id"
        ]]
    elif 'direction' in data.columns:
        data = data[["url", "formula", "factor_id", "direction", "plot"]]
    else:
        data = data[["url", "formula", "factor_id", "plot"]]

    return display(HTML(data.to_html(escape=False, index=False)))
    #return display(HTML(results.to_html(escape=False)))


def _get_font(size=26):
    """自适应加载清晰粗体字体"""
    try:
        # Pillow 10+ 原生支持直接指定 size
        return ImageFont.load_default(size=size)
    except Exception:
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _fast_merge_images_with_labels(split_plots,
                                   save_path,
                                   banner_height=60,
                                   overwrite=True):
    """
    在顶部为 train/val/recent 各自绘制专属色彩的标识横幅
    """
    if not overwrite and os.path.exists(save_path):
        return save_path
    labels = ['train', 'val', 'recent']
    colors = {
        'train': '#1F4E79',  # 深蓝色
        'val': '#2E75B6',  # 科技蓝
        'recent': '#C00000'  # 醒目红
    }
    valid_items = []
    for s in labels:
        p = split_plots.get(s)
        if p and os.path.exists(p):
            valid_items.append((s, Image.open(p)))
    if not valid_items:
        return None
    target_height = valid_items[0][1].height
    resized_imgs = [(s, img if img.height == target_height else img.resize(
        (int(img.width * (target_height / img.height)),
         target_height), Image.Resampling.BILINEAR)) for s, img in valid_items]
    total_width = sum(img.width for _, img in resized_imgs)
    total_height = target_height + banner_height
    # 1. 创建带顶部横幅的新画布
    merged_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(merged_img)
    font = _get_font(size=26)
    x_offset = 0
    for s, img in resized_imgs:
        # 2. 绘制该周期的顶部色块
        bg_color = colors.get(s, '#333333')
        draw.rectangle([x_offset, 0, x_offset + img.width, banner_height],
                       fill=bg_color)
        # 3. 居中绘制醒目的文字标识
        title_text = f"【 {s.upper()} SET 】"
        bbox = draw.textbbox((0, 0), title_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x_offset + (img.width - text_w) // 2
        text_y = (banner_height - text_h) // 2
        draw.text((text_x, text_y), title_text, fill="white", font=font)
        # 4. 将原图拼接在标识栏下方
        merged_img.paste(img, (x_offset, banner_height))
        x_offset += img.width
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged_img.save(save_path, format="PNG", compress_level=1)
    return save_path


def convert_to_merged_summary(draft_data, max_workers=8, overwrite=True):
    records = []
    tasks = []
    grouped = list(draft_data.groupby('factor_id', sort=False))
    for factor_id, group in grouped:
        first_row = group.iloc[0]
        split_plots = dict(zip(group['split'], group['plot']))

        valid_paths = [
            p for p in split_plots.values() if p and isinstance(p, str)
        ]
        merged_path = None
        if valid_paths:
            base_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(valid_paths[0])))
            save_dir = os.path.join(base_dir, "merged_plots")
            merged_path = os.path.join(save_dir, f"{factor_id}_merged.png")

            # 需要生成或覆盖
            if overwrite or not os.path.exists(merged_path):
                tasks.append((split_plots, merged_path))
        records.append({
            'factor_id':
            factor_id,
            'direction':
            first_row.get('direction'),
            'formula':
            first_row.get('formula'),
            'source':
            first_row.get('source'),
            'plot':
            merged_path if merged_path else
            (valid_paths[0] if valid_paths else None)
        })
    # 多线程并行渲染带标识的大图
    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_fast_merge_images_with_labels, plots, out, 60,
                                overwrite) for plots, out in tasks
            ]
            for f in futures:
                f.result()
    cols = ['factor_id', 'direction', 'formula', 'source']
    return pd.DataFrame(records)[cols]


import numpy as np


def compare_years_factors(method,
                          task_id,
                          left_instruments,
                          right_instruments,
                          period,
                          years=None,
                          filename=None,
                          condition=None,
                          category='annual'):
    if not years:
        raise ValueError('years 不能为空，例如 [2021, 2022, 2023, 2024, 2025]')
    years = sorted({int(year) for year in years})
    year_count = len(years)
    defaults = {
        'min_abs_total_ic': 0.01,
        'min_abs_ic_mean': 0.02,
        'positive_ret_years': int(np.ceil(year_count * 0.8)),
        'valid_ic_years': int(np.ceil(year_count * 0.8)),
        'same_ic_sign_years': int(np.ceil(year_count * 0.8)),
        'both_ret_pass_years': int(np.ceil(year_count * 0.6)),
        'both_ic_pass_years': int(np.ceil(year_count * 0.6)),
        'require_same_ic_direction': True
    }
    unknown = set(condition or {}).difference(defaults)
    if unknown:
        raise ValueError(f'未知筛选参数: {sorted(unknown)}')
    settings = dict(defaults)
    settings.update(condition or {})

    year_data = {
        year:
        fetch_data4(method=method,
                    task_id=task_id,
                    instruments=left_instruments,
                    windows=f'year_{year}',
                    period=period,
                    filename=filename,
                    category=category)
        for year in years
    }
    empty_years = [year for year, frame in year_data.items() if frame.empty]
    if empty_years:
        raise ValueError(f'以下年份没有加载到数据: {empty_years}')

    annual = pd.concat(
        [frame.copy().assign(year=year) for year, frame in year_data.items()],
        ignore_index=True)
    annual = _check_years_data(annual, years, left_instruments,
                               right_instruments)
    annual = _level0(annual, settings['min_abs_total_ic'],
                     settings['min_abs_ic_mean'])
    asset_summary = _level1(annual)
    asset_summary = _level2(annual, asset_summary, years,
                            settings['positive_ret_years'],
                            settings['valid_ic_years'],
                            settings['same_ic_sign_years'])
    screening, paired_year = _level3(annual, asset_summary, years,
                                     left_instruments, right_instruments,
                                     settings['both_ret_pass_years'],
                                     settings['both_ic_pass_years'],
                                     settings['require_same_ic_direction'])
    screening = screening.sort_values([
        'final_pass', 'both_ret_pass_years', 'both_ic_pass_years',
        'weakest_mean_abs_total_ic'
    ],
                                      ascending=False).reset_index(drop=True)
    return {
        'screening': screening,
        'passed': screening.loc[screening['final_pass']].copy(),
        'annual': annual,
        'asset_summary': asset_summary,
        'paired_year': paired_year,
        'condition': settings
    }


def _check_years_data(annual, years, left_instrument, right_instrument):
    """Validate and normalize annual factor evaluation rows."""
    required = {
        'expression', 'instrument', 'year', 'total_ret', 'total_ic', 'ic_mean'
    }
    missing = required.difference(annual.columns)
    if missing:
        raise ValueError(f'年度结果缺少字段: {sorted(missing)}')
    if annual.empty:
        raise ValueError('没有加载到年度绩效数据')

    annual = annual.copy()
    annual['instrument'] = annual['instrument'].astype(
        str).str.strip().str.lower()
    annual['year'] = pd.to_numeric(annual['year'], errors='coerce')
    for column in [
            'total_ret', 'total_ic', 'ic_mean', 'ann_sharpe', 'calmar',
            'max_dd', 'avg_ret'
    ]:
        if column in annual.columns:
            annual[column] = pd.to_numeric(annual[column], errors='coerce')

    expected_instruments = {
        str(left_instrument).strip().lower(),
        str(right_instrument).strip().lower()
    }
    actual_instruments = set(annual['instrument'].dropna().unique())
    missing_instruments = expected_instruments.difference(actual_instruments)
    if missing_instruments:
        raise ValueError(f'年度结果缺少品种: {sorted(missing_instruments)}')

    actual_years = set(annual['year'].dropna().astype(int).unique())
    missing_years = set(years).difference(actual_years)
    if missing_years:
        raise ValueError(f'年度结果缺少年份: {sorted(missing_years)}')

    duplicated = annual.duplicated(['expression', 'instrument', 'year'],
                                   keep=False)
    if duplicated.any():
        examples = annual.loc[duplicated,
                              ['expression', 'instrument', 'year']].head(10)
        print('同一因子、品种和年度存在重复记录:\n'
              f'{examples.to_string(index=False)}')
    annual = annual.drop_duplicates(
        subset=['expression', 'instrument', 'year'])
    return annual


def _level0(annual, min_abs_total_ic, min_abs_ic_mean):
    """Mark whether each annual row passes return and absolute-IC gates."""
    annual = annual.copy()
    annual['abs_total_ic'] = annual['total_ic'].abs()
    annual['abs_ic_mean'] = annual['ic_mean'].abs()
    annual['ret_pass'] = annual['total_ret'] > 0
    annual['ic_pass'] = ((annual['abs_total_ic'] >= min_abs_total_ic)
                         & (annual['abs_ic_mean'] >= min_abs_ic_mean))
    # These columns only check stability; they never change factor direction.
    annual['ic_positive'] = annual['total_ic'] > 0
    annual['ic_negative'] = annual['total_ic'] < 0
    return annual


def _level1(annual):
    """Summarize multi-year persistence for each factor and instrument."""
    asset_summary = annual.groupby(
        ['expression', 'instrument'],
        as_index=False).agg(year_count=('year', 'nunique'),
                            positive_ret_years=('ret_pass', 'sum'),
                            valid_ic_years=('ic_pass', 'sum'),
                            positive_ic_sign_years=('ic_positive', 'sum'),
                            negative_ic_sign_years=('ic_negative', 'sum'),
                            mean_abs_total_ic=('abs_total_ic', 'mean'),
                            mean_abs_ic_mean=('abs_ic_mean', 'mean'),
                            mean_total_ret=('total_ret', 'mean'))
    asset_summary['same_ic_sign_years'] = asset_summary[[
        'positive_ic_sign_years', 'negative_ic_sign_years'
    ]].max(axis=1)
    asset_summary['dominant_ic_sign'] = np.where(
        asset_summary['positive_ic_sign_years']
        >= asset_summary['negative_ic_sign_years'], 1, -1)
    return asset_summary


def _level2(annual, asset_summary, years, positive_ret_years, valid_ic_years,
            same_ic_sign_years):
    """Apply latest-year and per-instrument persistence requirements."""
    last_year = max(years)
    last_year_result = annual.loc[annual['year'] == last_year, [
        'expression', 'instrument', 'total_ret', 'abs_total_ic', 'abs_ic_mean',
        'ret_pass', 'ic_pass'
    ]].copy().rename(
        columns={
            'total_ret': 'last_total_ret',
            'abs_total_ic': 'last_abs_total_ic',
            'abs_ic_mean': 'last_abs_ic_mean',
            'ret_pass': 'last_ret_pass',
            'ic_pass': 'last_ic_pass'
        })
    asset_summary = asset_summary.merge(last_year_result,
                                        on=['expression', 'instrument'],
                                        how='left',
                                        validate='one_to_one')
    asset_summary['asset_pass'] = (
        (asset_summary['year_count'] == len(years))
        & (asset_summary['positive_ret_years'] >= positive_ret_years)
        & (asset_summary['valid_ic_years'] >= valid_ic_years)
        & (asset_summary['same_ic_sign_years'] >= same_ic_sign_years)
        & asset_summary['last_ret_pass'].fillna(False)
        & asset_summary['last_ic_pass'].fillna(False))
    return asset_summary


def _level3(annual,
            asset_summary,
            years,
            left_instrument,
            right_instrument,
            both_ret_pass_years,
            both_ic_pass_years,
            require_same_ic_direction=True):
    """Require both instruments to pass separately and in the same years."""
    left_instrument = str(left_instrument).strip().lower()
    right_instrument = str(right_instrument).strip().lower()
    source_columns = [
        'expression', 'year', 'instrument', 'ret_pass', 'ic_pass', 'total_ret',
        'abs_total_ic', 'abs_ic_mean'
    ]
    pair_source = annual[source_columns]

    def instrument_rows(instrument, prefix):
        return pair_source.loc[pair_source['instrument'] == instrument].drop(
            columns='instrument').rename(
                columns={
                    'ret_pass': f'{prefix}_ret_pass',
                    'ic_pass': f'{prefix}_ic_pass',
                    'total_ret': f'{prefix}_total_ret',
                    'abs_total_ic': f'{prefix}_abs_total_ic',
                    'abs_ic_mean': f'{prefix}_abs_ic_mean'
                })

    paired_year = instrument_rows(left_instrument,
                                  'left').merge(instrument_rows(
                                      right_instrument, 'right'),
                                                on=['expression', 'year'],
                                                how='inner',
                                                validate='one_to_one')
    paired_year['both_ret_pass'] = (
        paired_year['left_ret_pass'].fillna(False).astype(bool)
        & paired_year['right_ret_pass'].fillna(False).astype(bool))
    paired_year['both_ic_pass'] = (
        paired_year['left_ic_pass'].fillna(False).astype(bool)
        & paired_year['right_ic_pass'].fillna(False).astype(bool))

    pair_summary = paired_year.groupby('expression', as_index=False).agg(
        paired_year_count=('year', 'nunique'),
        both_ret_pass_years=('both_ret_pass', 'sum'),
        both_ic_pass_years=('both_ic_pass', 'sum'),
        left_mean_total_ret=('left_total_ret', 'mean'),
        right_mean_total_ret=('right_total_ret', 'mean'),
        left_mean_abs_total_ic=('left_abs_total_ic', 'mean'),
        right_mean_abs_total_ic=('right_abs_total_ic', 'mean'))
    pair_summary['weakest_mean_total_ret'] = pair_summary[[
        'left_mean_total_ret', 'right_mean_total_ret'
    ]].min(axis=1)
    pair_summary['weakest_mean_abs_total_ic'] = pair_summary[[
        'left_mean_abs_total_ic', 'right_mean_abs_total_ic'
    ]].min(axis=1)
    pair_summary['joint_pass'] = (
        (pair_summary['paired_year_count'] == len(years))
        & (pair_summary['both_ret_pass_years'] >= both_ret_pass_years)
        & (pair_summary['both_ic_pass_years'] >= both_ic_pass_years))

    def asset_rows(instrument, prefix):
        columns = [
            'expression', 'asset_pass', 'dominant_ic_sign',
            'positive_ret_years', 'valid_ic_years', 'same_ic_sign_years',
            'last_total_ret', 'last_abs_total_ic', 'last_abs_ic_mean'
        ]
        return asset_summary.loc[asset_summary['instrument'] == instrument,
                                 columns].rename(
                                     columns={
                                         column: f'{prefix}_{column}'
                                         for column in columns
                                         if column != 'expression'
                                     })

    asset_pair = asset_rows(left_instrument,
                            'left').merge(asset_rows(right_instrument,
                                                     'right'),
                                          on='expression',
                                          how='inner',
                                          validate='one_to_one')
    asset_pair['both_asset_pass'] = (
        asset_pair['left_asset_pass'].fillna(False).astype(bool)
        & asset_pair['right_asset_pass'].fillna(False).astype(bool))
    asset_pair['same_ic_direction'] = (asset_pair['left_dominant_ic_sign'] ==
                                       asset_pair['right_dominant_ic_sign'])

    screening = pair_summary.merge(asset_pair,
                                   on='expression',
                                   how='inner',
                                   validate='one_to_one')
    direction_pass = (screening['same_ic_direction']
                      if require_same_ic_direction else True)
    screening['final_pass'] = (screening['joint_pass']
                               & screening['both_asset_pass']
                               & direction_pass)
    return screening, paired_year


def _merge_annual_images_with_labels(window_plots,
                                     save_path,
                                     window_order,
                                     columns=2,
                                     banner_height=60,
                                     cell_width=1600,
                                     overwrite=True,
                                     png_colors=128,
                                     png_compress_level=6):
    """Merge full-period and annual plots into a labelled grid."""
    if not overwrite and os.path.exists(save_path):
        return save_path
    valid_items = [
        (window, window_plots.get(window)) for window in window_order
        if window_plots.get(window) and os.path.exists(window_plots[window])
    ]
    if not valid_items:
        return None
    if not isinstance(columns, int) or columns < 1:
        raise ValueError('columns 必须是正整数')
    if png_colors is not None and not 2 <= int(png_colors) <= 256:
        raise ValueError('png_colors 必须是 2 到 256，或设为 None')
    if not 0 <= int(png_compress_level) <= 9:
        raise ValueError('png_compress_level 必须是 0 到 9')
    # Keep this function self-contained: remote notebooks may import or copy
    # only this function without the module-level compatibility helpers.
    resampling = getattr(Image, 'Resampling', Image)
    bilinear_filter = resampling.BILINEAR

    images = []
    for window, path in valid_items:
        with Image.open(path) as source:
            images.append((window, source.convert('RGB').copy()))
    first_width, first_height = images[0][1].size
    target_width = min(int(cell_width),
                       first_width) if cell_width else first_width
    target_height = max(1, round(first_height * target_width / first_width))
    rows = int(np.ceil(len(images) / columns))
    canvas = Image.new('RGB', (target_width * columns,
                               (target_height + banner_height) * rows),
                       'white')
    draw = ImageDraw.Draw(canvas)
    font = _get_font(size=26)
    colors = [
        '#1F4E79', '#2E75B6', '#548235', '#BF9000', '#C65911', '#A61C00',
        '#7030A0', '#44546A'
    ]
    for index, (window, source) in enumerate(images):
        row, column = divmod(index, columns)
        x = column * target_width
        y = row * (target_height + banner_height)
        scale = min(target_width / source.width, target_height / source.height)
        resized = source.resize((max(1, round(
            source.width * scale)), max(1, round(source.height * scale))),
                                bilinear_filter)
        image_x = x + (target_width - resized.width) // 2
        image_y = y + banner_height + (target_height - resized.height) // 2
        draw.rectangle([x, y, x + target_width, y + banner_height],
                       fill=colors[index % len(colors)])
        title = window.upper()
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), title, font=font)
        else:
            text_width, text_height = draw.textsize(title, font=font)
            bbox = (0, 0, text_width, text_height)
        text_x = x + (target_width - (bbox[2] - bbox[0])) // 2
        text_y = y + (banner_height - (bbox[3] - bbox[1])) // 2
        draw.text((text_x, text_y), title, fill='white', font=font)
        canvas.paste(resized, (image_x, image_y))
        source.close()
        resized.close()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if png_colors is None:
        output_image = canvas
    else:
        palette = getattr(Image, 'Palette', Image)
        output_image = canvas.convert('P',
                                      palette=palette.ADAPTIVE,
                                      colors=int(png_colors))
    output_image.save(save_path,
                      format='PNG',
                      compress_level=int(png_compress_level))
    if output_image is not canvas:
        output_image.close()
    canvas.close()
    return save_path


def _write_annual_plots_html(window_plots,
                             save_path,
                             window_order,
                             title,
                             columns=2,
                             overwrite=True):
    """Create a lightweight HTML grid which references the original plots."""
    if not overwrite and os.path.exists(save_path):
        return save_path
    if not isinstance(columns, int) or columns < 1:
        raise ValueError('columns 必须是正整数')
    valid_items = [
        (window, window_plots.get(window)) for window in window_order
        if window_plots.get(window) and os.path.exists(window_plots[window])
    ]
    if not valid_items:
        return None

    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    figures = []
    for window, image_path in valid_items:
        relative_path = os.path.relpath(image_path,
                                        output_dir).replace(os.sep, '/')
        figures.append('<figure>'
                       f'<figcaption>{escape(window.upper())}</figcaption>'
                       f'<a href="{escape(relative_path, quote=True)}">'
                       f'<img src="{escape(relative_path, quote=True)}" '
                       'loading="lazy" decoding="async"></a>'
                       '</figure>')
    html = f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
  body {{ margin: 0; padding: 16px; background: #181a1f; color: #f3f3f3;
          font-family: Arial, sans-serif; }}
  h1 {{ margin: 0 0 16px; font-size: 18px; word-break: break-all; }}
  .grid {{ display: grid; grid-template-columns: repeat({columns}, minmax(0, 1fr));
           gap: 14px; align-items: start; }}
  figure {{ margin: 0; min-width: 0; background: #fff; border-radius: 4px;
            overflow: hidden; }}
  figcaption {{ padding: 10px; color: #fff; background: #1f4e79;
                text-align: center; font-weight: bold; }}
  img {{ display: block; width: 100%; height: auto; object-fit: contain; }}
  @media (max-width: 1100px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
<main class="grid">{''.join(figures)}</main>
</body>
</html>'''
    with open(save_path, 'w', encoding='utf-8') as stream:
        stream.write(html)
    return save_path


def merge_annual_factor_plots(method,
                              task_id,
                              instruments,
                              period,
                              expressions,
                              years,
                              category='annual',
                              full_window='three_year',
                              columns=2,
                              cell_width=1600,
                              max_workers=4,
                              overwrite=True,
                              annual_root=None,
                              output_format='png',
                              png_colors=128,
                              png_compress_level=6):
    """Load and merge each selected factor's full-period and yearly plots.

    ``expressions`` accepts an iterable of formulas or a DataFrame containing
    ``expression``/``formula``. ``annual_root`` may point directly to the
    directory containing ``three_year`` and ``year_YYYY`` folders.
    """
    if expressions is None:
        raise ValueError('expressions 不能为空')
    if isinstance(expressions, pd.DataFrame):
        formula_column = (
            'expression' if 'expression' in expressions.columns else
            'formula' if 'formula' in expressions.columns else None)
        if formula_column is None:
            raise ValueError('DataFrame 需要 expression 或 formula 字段')
        formulas = expressions[formula_column].dropna().astype(str).tolist()
    elif isinstance(expressions, str):
        formulas = [expressions]
    else:
        formulas = [
            str(expression) for expression in expressions
            if pd.notna(expression)
        ]
    formulas = list(dict.fromkeys(formulas))
    if not formulas:
        raise ValueError('没有有效的因子表达式')
    if not years:
        raise ValueError('years 不能为空')
    output_format = str(output_format).strip().lower()
    if output_format not in {'png', 'html'}:
        raise ValueError("output_format 只能是 'png' 或 'html'")
    years = sorted({int(year) for year in years})
    window_order = ([full_window] if full_window else
                    []) + [f'year_{year}' for year in years]

    if annual_root is not None:
        roots = [Path(annual_root)]
    else:
        relative = (Path(method) / str(instruments) / 'rulex' / str(task_id) /
                    f'nxt1_ret_{period}h' / str(category))
        roots = [Path(base_path) / relative, Path('records') / relative]
    expanded_roots = []
    for root in roots:
        expanded_roots.extend([root, root / 'recent'])
    roots = list(dict.fromkeys(expanded_roots))
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        raise FileNotFoundError('未找到年度图片目录: ' +
                                ', '.join(str(root) for root in roots))

    output_root = existing_roots[0] / ('merged_year_html' if output_format
                                       == 'html' else 'merged_year_plots')
    tasks = []
    records = []
    for formula in formulas:
        factor_id = str(create_id(generate_simple_id(formula)))
        window_plots = {}
        for window in window_order:
            for root in existing_roots:
                factor_dir = root / window / factor_id
                candidates = [
                    factor_dir / 'comparison_plot.png',
                    factor_dir / 'evaluation_plot.png'
                ]
                plot = next((path for path in candidates if path.exists()),
                            None)
                if plot is not None:
                    window_plots[window] = str(plot)
                    break
        available = [
            window for window in window_order if window in window_plots
        ]
        missing = [
            window for window in window_order if window not in window_plots
        ]
        suffix = 'html' if output_format == 'html' else 'png'
        merged_path = output_root / f'{factor_id}_annual_merged.{suffix}'
        if available and (overwrite or not merged_path.exists()):
            tasks.append((window_plots, str(merged_path), formula))
        records.append({
            'factor_id': factor_id,
            'formula': formula,
            'expression': formula,
            'plot': str(merged_path) if available else None,
            'available_windows': ','.join(available),
            'missing_windows': ','.join(missing),
            'complete': not missing
        })

    if tasks and output_format == 'html':
        for plots, output, formula in tasks:
            _write_annual_plots_html(plots,
                                     output,
                                     window_order,
                                     formula,
                                     columns=columns,
                                     overwrite=overwrite)
    elif tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_merge_annual_images_with_labels, plots,
                                output, window_order, columns, 60, cell_width,
                                overwrite, png_colors, png_compress_level)
                for plots, output, _ in tasks
            ]
            for future in futures:
                future.result()
    result = pd.DataFrame(records)
    result['file_size_mb'] = result['plot'].map(
        lambda path: round(os.path.getsize(path) / 1024**2, 3)
        if path and os.path.exists(path) else np.nan)
    return result
