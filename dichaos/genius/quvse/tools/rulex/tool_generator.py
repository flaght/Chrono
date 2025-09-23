import os, random, pdb, json
import pandas as pd
from utils.parsers import generate_markdown_table


def read_factor_groups(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_detailed_list(factor_groups, factor_descriptions):
    """
    根据基础因子描述，为具体的因子列名构建一个详细的字典列表。

    参数:
    factor_groups (dict): 一个字典，键是基础因子名，值是具体的列名列表。
    factor_descriptions (list): 一个字典列表，每个字典包含'Field'和'Description'。

    返回:
    list: 一个新的字典列表，每个字典包含具体的'Field'和其对应的'Description'。
    """
    # 步骤 1: 创建一个高效的查找映射
    # 键是基础因子名 (e.g., 'cr029'), 值是描述
    description_map = {
        item['Field']: item['Description']
        for item in factor_descriptions
    }

    # 步骤 2: 遍历分组并构建新的列表
    new_data_list = []
    # 遍历 factor_groups 中的每一项 (e.g., 'cj002': ['cj002_10_15_0', ...])
    for base_factor, specific_fields in factor_groups.items():
        # 查找当前基础因子的描述
        # 使用 .get() 以防某个因子在描述列表中不存在
        description = description_map.get(base_factor,
                                          "未找到描述 (Description not found)")

        # 为每个具体的列名创建一个新的字典
        for field_name in specific_fields:
            new_data_list.append({
                'Field': field_name,
                'Description': description
            })

    return new_data_list


class ToolGenerator(object):

    def __init__(self):
        self.base_path = os.path.join("records/factors/")

        #features_pd = self.features_basic

        self.features_basic = self._read_pd(
            file_name="basic_fields_dependencies.csv",
            filed_names=['Field', 'Description'])

        self.groups_basic = read_factor_groups(
            os.path.join(self.base_path, "basic_fields_dependencies.json"))

        self.features_basic_pd = build_detailed_list(
            factor_groups=self.groups_basic,
            factor_descriptions=self.features_basic)

        self.features_level2 = self._read_pd(
            file_name="level2_fields_dependencies.csv",
            filed_names=['Field', 'Description'])

        self.singal = self._read_pd(
            file_name="signal_dependencies.csv",
            filed_names=['SignalName', 'StrategyExplanation','KeyParameters'])

        self.holding = self._read_pd(
            file_name="holding_dependencies.csv",
            filed_names=['StrategyName', 'StrategyExplanation','KeyParameters'])

        self.expression = self._read_pd(
            file_name="expression_dependencies.csv",
            filed_names=['Expression', 'Description'])

    def _read_pd(self, file_name, filed_names):
        data = pd.read_csv(os.path.join(self.base_path, file_name))
        df1 = data[filed_names].to_dict(orient='records')
        return df1

    def fetch_random(self, data, k):
        random_data = random.sample(data, k)
        markdwon_str = generate_markdown_table(random_data)
        return markdwon_str

    def create_tools(self, category, k):
        if category == 'level2':
            features_pd = self.features_level2
            features = self.fetch_random(data=features_pd, k=k * 5),
        elif category == 'basic':
            features_pd = self.features_basic_pd
            features = self.fetch_random(data=features_pd, k=k * 5)

        singal_pd = self.singal
        holding_pd = self.holding

        expression = self.expression

        return {
            'features': features,
            'signal': self.fetch_random(data=singal_pd, k=3),
            'holding': self.fetch_random(data=holding_pd, k=3),
            'expression': self.fetch_random(data=expression, k=6)
        }
