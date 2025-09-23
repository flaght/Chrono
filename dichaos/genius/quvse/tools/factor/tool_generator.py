import os, random, pdb, re, json
import pandas as pd
from ultron.sentry.api import *
from ultron.factor.genetic.geneticist.operators import calc_factor
from utils.parsers import generate_markdown_table


def read_factor_groups(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def reshape_operator_list_loop(data: list) -> dict:
    """
    使用for循环将操作符列表重塑为以Operator为键的字典。
    """
    reshaped_dict = {}

    # 遍历列表中的每一个字典
    for item in data:
        # 检查 'Operator' 键是否存在，以避免KeyError
        if 'Operator' in item:
            # 弹出 'Operator' 键及其值，作为新字典的键
            operator_key = item.pop('Operator')

            # 将剩余的字典作为新字典的值
            reshaped_dict[operator_key] = item

    return reshaped_dict


def extract_mathrm_operators(text: str) -> list[str]:
    """
    使用正则表达式从包含 LaTeX `\mathrm{}` 标记的字符串中提取所有算子名称。

    Args:
        text: 包含 LaTeX 标记的输入字符串。

    Returns:
        一个包含所有提取出的算子名称的列表。
    """
    # 正则表达式，用于查找 \mathrm{...} 结构并捕获其中的内容
    # 捕获组 ([a-zA-Z0-9_]+) 会匹配由字母、数字和下划线组成的算子名
    regex = r"\\mathrm\{([a-zA-Z0-9_]+)\}"

    # re.findall 会返回所有捕获组匹配到的字符串列表
    operators = re.findall(regex, text)

    return operators


def invert_factor_dict_comprehension(data: dict) -> dict:
    """
    使用字典推导式将因子字典反转并展平。
    """
    return {
        factor: base_factor
        for base_factor, specific_factors in data.items()
        for factor in specific_factors
    }


class ToolGenerator(object):

    def __init__(self):
        self.base_path = os.path.join("records/factors/")

        self.features_basic = self._read_pd(
            file_name="basic_fields_dependencies.csv",
            filed_names=['Field', 'Description'])

        self.groups_basic = read_factor_groups(
            os.path.join(self.base_path, "basic_fields_dependencies.json"))

        self.features_level2 = self._read_pd(
            file_name="level2_fields_dependencies.csv",
            filed_names=['Field', 'Description'])

        self.singal = self._read_pd(
            file_name="signal_dependencies.csv",
            filed_names=['SignalName', 'StrategyExplanation'])

        self.holding = self._read_pd(
            file_name="holding_dependencies.csv",
            filed_names=['StrategyName', 'StrategyExplanation'])

        self.expression = self._read_pd(
            file_name="expression_dependencies.csv",
            filed_names=['Expression', 'Description', 'Operator'])

        ## 转字典
        self.features_basic_dict = dict(
            zip([fb['Field'] for fb in self.features_basic],
                [fb['Description'] for fb in self.features_basic]))
        self.features_level2_dict = dict(
            zip([fb['Field'] for fb in self.features_level2],
                [fb['Description'] for fb in self.features_level2]))

        self.operator_dict = reshape_operator_list_loop(self.expression)

        self.comp_basic_dict = invert_factor_dict_comprehension(
            self.groups_basic)
        self.comp_level2_dict = self.features_level2_dict

        base_features = pd.DataFrame([self.comp_basic_dict]).T.reset_index()
        base_features.columns = ['factor', 'Field']
        base_features = base_features.merge(
            pd.DataFrame(self.features_basic),
            on=['Field']).drop(['Field'],
                               axis=1).rename(columns={'factor': 'Field'})
        self.base_features = base_features.to_dict(orient='records')

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
        elif category == 'basic':
            features_pd = self.features_basic

        singal_pd = self.singal
        holding_pd = self.holding

        expression = self.expression

        return {
            'features': self.fetch_random(data=features_pd, k=k * 5),
            'signal': self.fetch_random(data=singal_pd, k=3),
            'holding': self.fetch_random(data=holding_pd, k=3),
            'expression': self.fetch_random(data=expression, k=6)
        }

    def create_tools1(self, category, k):
        if category == 'level2':
            features_pd = self.features_level2
        elif category == 'basic':
            features_pd = self.base_features

        singal_pd = self.singal
        holding_pd = self.holding

        expression = self.expression

        return {
            'features': self.fetch_random(data=features_pd, k=k * 5),
            'signal': self.fetch_random(data=singal_pd, k=3),
            'holding': self.fetch_random(data=holding_pd, k=3),
            'expression': self.fetch_random(data=expression, k=6)
        }

    def expression_disassembly(self, category, expression):
        if category == 'level2':
            features_dict = self.comp_level2_dict
        elif category == 'basic':
            features_dict = self.comp_basic_dict
        features = eval(expression).fields
        operators = extract_mathrm_operators(eval(expression).__str__())
        ### 算子
        operators_list = []
        for op1 in operators:
            if op1 in self.operator_dict:
                vals = self.operator_dict[op1]
                vals['Operator'] = op1
                operators_list.append(vals)
            elif op1.upper() in self.operator_dict:
                vals = self.operator_dict[op1.upper()]
                vals['Operator'] = op1.upper()
                operators_list.append(vals)
        ### 特征
        features_list = []
        for ft1 in features:
            ft2 = features_dict[ft1]
            features_list.append({
                'Field': ft2,
                'Description': self.features_basic_dict[ft2]
            })

        return {
            'features': generate_markdown_table(features_list),
            'expression': generate_markdown_table(operators_list)
        }

    def expressions_disassembly(self, category, expressions):
        if category == 'level2':
            features_dict = self.comp_level2_dict
        elif category == 'basic':
            features_dict = self.comp_basic_dict

        operators_list = []
        features_list = []
        for expression in expressions:
            features = eval(expression).fields
            operators = extract_mathrm_operators(eval(expression).__str__())

            for ft1 in features:
                ft2 = features_dict[ft1]
                features_list.append({
                    'Field':
                    ft1,
                    'Description':
                    self.features_basic_dict[ft2]
                })

            for op1 in operators:
                if op1 in self.operator_dict:
                    vals = self.operator_dict[op1]
                    vals['Operator'] = op1
                    operators_list.append(vals)
                elif op1.upper() in self.operator_dict:
                    vals = self.operator_dict[op1.upper()]
                    vals['Operator'] = op1.upper()
                    operators_list.append(vals)

        return {
            'features': generate_markdown_table(features_list),
            'expression': generate_markdown_table(operators_list)
        }
