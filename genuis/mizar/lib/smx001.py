import pandas as pd
import io, re
from lib.iux002 import generate_simple_id
from lumina.genetic.util import create_id
from IPython.display import display, HTML
from kdutils.macro2 import *


def parse_factor_file(file_path):
    """
    自动识别并解析因子结果文件。
    兼容格式1（带 | 的表格）和格式2（冒号键值对）。
    """
    # 1. 读取文件内容
    if not os.path.exists(file_path):
        return pd.DataFrame, None

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. 提取元数据（Metadata，如 Expression, Name, Window 等）
    metadata = {}
    meta_keys = [
        'Expression', 'Name', 'Roll Window', 'Resampling Window',
        'Holding Profit'
    ]
    for line in text.split('\n'):
        line = line.strip()
        for mk in meta_keys:
            if line.startswith(mk):
                # 按照冒号分割提取值
                val = line.split(':', 1)[1].strip()
                metadata[mk] = val

    # 3. 核心解析逻辑：判断属于哪种格式
    if '|' in text and 'Metric' in text:
        # ======= 格式1：表格格式 (提取 ims) =======
        table_lines = [line for line in text.split('\n') if '|' in line]
        table_text = '\n'.join(table_lines)
        df = pd.read_csv(io.StringIO(table_text), sep='|')

        # 清洗列名和字符串前后的空格
        df.columns = df.columns.str.strip()
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()

        # 核心需求：只保留 Metric 和 ims（左边的那列）
        if 'ims' in df.columns:
            df = df[['Metric', 'ims']]
            df.rename(columns={'ims': 'Value'}, inplace=True)  # 统一重命名为 Value

    else:
        # ======= 格式2：键值对格式 =======
        metrics_data = []
        for line in text.split('\n'):
            line = line.strip()
            # 跳过空行、分隔符(---)以及刚才已经提过的元数据
            if not line or line.startswith('---'):
                continue

            # 如果行内包含冒号，且不是我们已提取的元数据
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                val = val.strip()

                if key not in meta_keys:
                    metrics_data.append({'Metric': key, 'Value': val})

        df = pd.DataFrame(metrics_data)

    # 4. 格式转换：把百分号去掉变小数，纯数字转为 float
    def convert_to_numeric(val):
        if pd.isna(val):
            return val
        if isinstance(val, str) and val.endswith('%'):
            try:
                return float(val.replace('%', '')) / 100.0
            except:
                pass
        try:
            return float(val)
        except ValueError:
            return val

    # 应用转换并将 Metric 设为索引
    if not df.empty:
        df['Value'] = df['Value'].apply(convert_to_numeric)
        df.set_index('Metric', inplace=True)

    return df, metadata


def fetch_chosen(method, instruments, task_id, period):
    basic_path = os.path.join(base_path, method, instruments, "rulex",
                              str(task_id), "nxt1_ret_{}h".format(str(period)))
    filename = os.path.join(basic_path, "draft.csv")
    chosen_data = pd.read_csv(filename)
    chosen_data['basic_path'] = basic_path
    return chosen_data


def make_clickable(val):
    return f'<a target="_blank" href="{val}">{val}</a>'


def create_path(x):
    c1 = str(x['source']) if x['category'] == 'p' else f"d{x['source']}"
    return os.path.join(x['basic_path'], c1, x['name'])

def create_image(x):
    c1 = str(x['source']) if x['category'] == 'p' else f"d{x['source']}"
    name = 'comparison_plot.png' if x['category'] == 'p' else 'evaluation_plot.png'
    return os.path.join(x['basic_path'], c1, x['name'], name)


def create_plot(method, instruments, period, task_id, names=[]):
    draft_data = fetch_chosen(method=method,
                              instruments=instruments,
                              task_id=task_id,
                              period=period)
    draft_data['name'] = draft_data['formula'].apply(
        lambda x: create_id(generate_simple_id(x)))
    draft_data['image_path'] = draft_data.apply(lambda x: create_image(x),
                                                axis=1)
    draft_data['image_path'] = draft_data['image_path'].apply(make_clickable)
    if len(names) > 0:
        draft_data = draft_data[draft_data.name.isin(names)]
    draft_data = draft_data.drop(
        ['direction', 'source', 'category', 'basic_path', 'name'], axis=1)
    return HTML(draft_data.to_html(escape=False))


def create_metrics(method, instruments, period, task_id):
    draft_data = fetch_chosen(method=method,
                              instruments=instruments,
                              task_id=task_id,
                              period=period)
    draft_data['name'] = draft_data['formula'].apply(
        lambda x: create_id(generate_simple_id(x)))
    draft_data['basic_path'] = draft_data.apply(lambda x: create_path(x),
                                                axis=1)
    draft_data[
        'metrics_path'] = draft_data['basic_path'] + '/performance_summary.txt'

    res = []
    for row in draft_data.itertuples():
        df, _ = parse_factor_file(row.metrics_path)
        if df.empty:
            continue
        df = df.T
        df['formula'] = row.formula
        df['name'] = row.name
        res.append(df)
    data = pd.concat(res).drop(['Win Rate', 'Profit/Loss Ratio'], axis=1)
    return data.rename(columns={'Avg Return (bps)':'avg_return',
                                'Total Return':'total_return',
                                'Sharpe Ratio':'sharpe',
                                'Ann Sharpe Ratio':'ann_sharpe',
                                'Max Drawdown':'max_drawdown',
                                'Calmar Ratio': 'calmar',
                                'IC Mean':'ic',
                                'ICIR':'icir',
                                'Mean Turnover': 'turnover',
                                'Factor Autocorr':'factor_autocorr',
                                'Return Autocorr': 'return_autocorr'})
