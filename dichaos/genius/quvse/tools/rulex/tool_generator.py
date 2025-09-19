import os, random, pdb
import pandas as pd
from utils.parsers import generate_markdown_table


class ToolGenerator(object):

    def __init__(self):
        self.base_path = os.path.join("records/factors/")
        self.features_basic = self._read_pd(
            file_name="basic_fields_dependencies.csv",
            filed_names=['Field', 'Description'])

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
