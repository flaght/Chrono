import re, json, xmltodict
import pandas as pd
from io import StringIO
from typing import List, Dict, Any, Optional

def convert_str_to_markdown_table(data_string: str) -> str:
    """
    将 'key:value,key:value' 格式的字符串转换为Markdown表格。

    Args:
        data_string: 输入的字符串。

    Returns:
        一个格式化的Markdown表格字符串。
    """
    # Markdown表格的表头
    markdown_header = "| 标识符 (Identifier) | 描述 (Description) |\n|---|---|\n"

    markdown_rows = []

    # 1. 按逗号和空格分割字符串，得到键值对列表
    #    strip() 用于去除首尾可能存在的空白
    pairs = data_string.strip().split(',')

    for pair in pairs:
        # 去除每个键值对周围的空白
        pair = pair.strip()
        if not pair:
            continue

        # 2. 按冒号分割键和值
        #    split(':', 1) 确保只在第一个冒号处分割，以防值中也包含冒号
        parts = pair.split(':', 1)

        if len(parts) == 2:
            identifier = parts[0].strip()
            description = parts[1].strip()

            # 3. 构建Markdown表格的一行
            markdown_rows.append(f"| {identifier} | {description} |")
        else:
            print(f"[Warning] Skipping invalid pair: {pair}")

    # 组合表头和所有行
    return markdown_header + "\n".join(markdown_rows)


def generate_markdown_table(data_list):
    """
    将一个字典列表转换为Markdown表格字符串。
    
    参数:
    data_list (list): 一个包含字典的列表，每个字典代表一行。
                      假设所有字典都有相同的键。

    返回:
    str: Markdown格式的表格字符串。
    """
    if not data_list:
        return "输入数据为空。"

    # 1. 提取表头 (从第一个字典的键)
    headers = list(data_list[0].keys())

    # 2. 构建表头行和分隔行
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join([":---"] * len(headers)) + " |"

    # 3. 构建数据行
    data_lines = []
    for item in data_list:
        # 按表头顺序提取值
        row_values = [str(item.get(h, '')) for h in headers]
        data_lines.append("| " + " | ".join(row_values) + " |")

    # 4. 组合所有部分
    markdown_table = "\n".join([header_line, separator_line] + data_lines)

    return markdown_table


class FinalAnswerParser:
    """
    一个专门用于解析LLM生成的、包含Markdown表格的最终答案的工具类。
    """

    def __init__(self):
        self.markdown_table_regex = re.compile(
            r"^\s*\|.*\|\s*\n\s*\|.*--.*\|\s*\n((?:\|.*\|\s*\n?)+)",
            re.MULTILINE)

    def convert_xml_to_json(xml_string: str, indent: int = 2) -> str:
        """
        将XML字符串快速转换为格式化的JSON字符串。

        Args:
            xml_string: 包含XML数据的字符串。
            indent: JSON输出的缩进级别。设置为None可输出无格式的单行JSON。

        Returns:
            一个格式化的JSON字符串。如果解析失败，则返回一个包含错误信息的JSON。
        """
        try:
            # 核心步骤：使用 xmltodict.parse 将XML解析为Python字典
            data_dict = xmltodict.parse(xml_string)

            # 可选步骤：对字典结构进行微调
            # 在这个例子中，xmltodict的默认转换结果已经很好了，
            # 但我们也可以在这里进行自定义处理，例如将 'variant' 列表提取出来。

            # 另一个核心步骤：使用 json.dumps 将字典转换为JSON字符串
            # ensure_ascii=False 确保中文字符能被正确显示，而不是被转义成 \uXXXX
            json_string = json.dumps(data_dict,
                                     indent=indent,
                                     ensure_ascii=False)

            return json_string

        except Exception as e:
            error_message = f"Failed to convert XML to JSON. Error: {e}"
            print(error_message)
            return json.dumps({"error": error_message}, indent=indent)

    def parse_markdown_table_to_json(
            self, markdown_text: str) -> Optional[List[Dict[str, Any]]]:
        match = self.markdown_table_regex.search(markdown_text)
        if not match:
            return None

        table_text = match.group(0)
        try:
            df = pd.read_csv(StringIO(table_text),
                             sep="|",
                             header=0,
                             skipinitialspace=True).dropna(axis=1,
                                                           how='all').iloc[1:]

            df.columns = [col.strip() for col in df.columns]
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            rename_map = {
                '因子表达式': 'expression',
                '解释': 'explanation',
                '原理': 'principle',
                '评分': 'score'
            }
            required_cols = [
                col for col in rename_map.keys() if col in df.columns
            ]
            if not required_cols: return None

            df = df[required_cols].rename(columns=rename_map)

            if 'score' in df.columns:
                df['score'] = pd.to_numeric(
                    df['score'], errors='coerce').fillna(0).astype(int)

            return df.to_dict(orient='records')
        except Exception as e:
            print(f"[Parser Error] 使用Pandas解析Markdown表格时出错: {e}")
            return None
