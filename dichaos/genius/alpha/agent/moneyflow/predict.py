import pdb
from agent.predictor import Predictor as BasePredictor
from agent.moneyflow.agent import Agent
from factors.calculator import create_moneyflow
from agent.indicator.model import FactorsList, FactorsGroup


class Predictor(BasePredictor):

    def __init__(self, symbol, memory_path, date):
        super(Predictor, self).__init__(symbol=symbol, agent_class=Agent)
        self.initialize_agent(memory_path=memory_path, date=date)

    def prepare_data(self, begin_date, end_date):

        self.specific_factors = create_moneyflow(begin_date=begin_date,
                                                 end_date=end_date,
                                                 codes=[self.symbol],
                                                 window=50)

    def create_group(self, date):
        """为给定日期创建已包含的FactorsGroup。"""
        specific_factors_list = self.create_model(date, self.specific_factors,
                                                  "基于日资金流向，计算生成的特征值",
                                                  FactorsList)
        factors_group = FactorsGroup(date=date, symbol=self.symbol)
        factors_group.factors_list.append(specific_factors_list)
        return factors_group
