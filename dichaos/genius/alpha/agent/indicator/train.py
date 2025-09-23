from agent.trainer import Trainer as BaseTrainer
from agent.indicator.agent import Agent
from factors.calculator import create_indictor, create_kline
from agent.indicator.model import FactorsList, FactorsGroup


class Trainer(BaseTrainer):

    def __init__(self,
                 symbol,
                 config_path,
                 memory_path,
                 date=None,
                 portfolio=None):
        super(Trainer, self).__init__(symbol=symbol,
                                      agent_class=Agent,
                                      portfolio=portfolio)
        self.initialize_agent(config_path=config_path,
                              memory_path=memory_path,
                              date=date)

    def prepare_data(self, begin_date, end_date):
        self.indicator_factors = create_indictor(begin_date=begin_date,
                                                 end_date=end_date,
                                                 codes=[self.symbol],
                                                 window=40)

        self.k1line_factors = create_kline(begin_date=begin_date,
                                           end_date=end_date,
                                           codes=[self.symbol],
                                           window=0)

        self.k5line_factors = create_kline(begin_date=begin_date,
                                           end_date=end_date,
                                           codes=[self.symbol],
                                           window=5)

    def create_group(self, date):
        """为给定日期创建已包含的FactorsGroup。"""
        indicator_factors_list = self.create_model(date,
                                                   self.indicator_factors,
                                                   "基于日频K线的技术指标", FactorsList)
        k1line_factors_list = self.create_model(date, self.k1line_factors,
                                                "基于日频的1天K线数据", FactorsList)

        k5line_factors_list = self.create_model(date, self.k5line_factors,
                                                "基于日频的5天K线数据", FactorsList)

        factors_group = FactorsGroup(date=date, symbol=self.symbol)
        factors_group.factors_list.append(indicator_factors_list)
        factors_group.factors_list.append(k1line_factors_list)
        factors_group.factors_list.append(k5line_factors_list)
        return factors_group
