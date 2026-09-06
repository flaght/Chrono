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


def make_clickable(val):
    return f'<a target="_blank" href="{val}">{val}</a>'


def extract_href(value):
    """兼容纯路径和已经生成的 HTML 超链接。"""
    text = str(value)

    match = re.search(r'''href=["']([^"']+)["']''', text)
    if match:
        return unescape(match.group(1))

    return unescape(text)


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
    else:
        data = data[["url", "formula", "plot", "factor_id"]]

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
