import pdb, os
from agent.predictor import Predictor as BasePredictor
from factors.calculator import create_heat
from agent.clouto.model import FactorsList, FactorsGroup
from agent.clouto.agent import Agent


class Predictor(BasePredictor):

    def __init__(self, symbol, memory_path, config_path, date):
        super(Predictor, self).__init__(symbol=symbol, agent_class=Agent)
        #self.initialize_agent(memory_path=memory_path, date=date)
        self.initialize_agent(memory_path=memory_path,
                              config_path=os.path.join(config_path,
                                                       self.agent_class.name),
                              date=date)

    def prepare_data(self, begin_date, end_date):
        self.xueqiu_factors = create_heat(begin_date=begin_date,
                                          end_date=end_date,
                                          codes=[self.symbol],
                                          window=100,
                                          category='xueqiu')
        self.guba_factors = create_heat(begin_date=begin_date,
                                        end_date=end_date,
                                        codes=[self.symbol],
                                        window=1000,
                                        category='guba')

    def create_group(self, date):
        """为给定日期创建已包含的FactorsGroup。"""
        xueqiu_factors_list = self.create_model(date, self.xueqiu_factors,
                                                "基于雪球热度的数据，计算生成的特征值",
                                                FactorsList)
        guba_factors_list = self.create_model(date, self.guba_factors,
                                              "基于股吧热度的数据，计算生成的特征值",
                                              FactorsList)

        factors_group = FactorsGroup(date=date, symbol=self.symbol)
        factors_group.factors_list.append(xueqiu_factors_list)
        factors_group.factors_list.append(guba_factors_list)
        return factors_group
