from agent.trainer import Trainer as BaseTrainer
from agent.posflow.agent import Agent
from factors.calculator import create_posflow
from agent.posflow.model import FactorsList, FactorsGroup


class Trainer(BaseTrainer):

    def __init__(self, symbol, config_path, memory_path, portfolio=None):
        super(Trainer, self).__init__(symbol=symbol,
                                      agent_class=Agent,
                                      portfolio=portfolio)
        self.initialize_agent(config_path=config_path, memory_path=memory_path)
        

    def prepare_data(self, begin_date, end_date):
        self.equal_factors = create_posflow(begin_date=begin_date,
                                            end_date=end_date,
                                            codes=[self.symbol],
                                            window=40,
                                            category='equal')
        self.weighted_factors = create_posflow(begin_date=begin_date,
                                               end_date=end_date,
                                               codes=[self.symbol],
                                               window=40,
                                               category='weighted')

    def create_group(self, date):
        """为给定日期创建已包含的FactorsGroup。"""
        equal_factors_list = self.create_model(date, self.equal_factors,
                                               "基于各个会员持仓等权相加合成的数据，计算生成的特征值",
                                               FactorsList)
        weight_factors_list = self.create_model(date, self.weighted_factors,
                                                "基于各个会员持仓加权相加合成的数据，计算生成的特征值",
                                                FactorsList)
        factors_group = FactorsGroup(date=date, symbol=self.symbol)
        factors_group.factors_list.append(equal_factors_list)
        factors_group.factors_list.append(weight_factors_list)
        return factors_group
