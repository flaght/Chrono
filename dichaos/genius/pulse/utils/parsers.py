import re, json, xmltodict
import pandas as pd
from io import StringIO
from typing import List, Dict, Any, Optional


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
