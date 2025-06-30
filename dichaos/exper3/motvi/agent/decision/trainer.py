import os
from abc import ABC, abstractmethod
from motvi.agent.trainer import Trainer as BaseTrainer
from motvi.agent.predictor import Predictor
from motvi.agent.decision.model import *
from motvi.agent.decision.agent import Agent


class Trainer(BaseTrainer):

    def __init__(self, symbol, base_path):
        super(Trainer, self).__init__(symbol=symbol, agent_class=Agent)
        self.initialize_agent(base_path)
        self.agent_pool = []

    def add_sub_agent(self, agent: Predictor):
        self.agent_pool.append(agent)

    ## 子策略准备的数据
    def prepare_data(self, begin_date, end_date):
        for predictor in self.agent_pool:
            predictor.prepare_data(begin_date=begin_date, end_date=end_date)

    def create_group(self, date):
        agents_group = AgentsGroup(date=date, symbol=self.symbol)
        for predictor in self.agent_pool:
            response = predictor.predict(date=date)
            result = AgentsResult(date=date,
                                  symbol=self.symbol,
                                  name=predictor.agent.name,
                                  desc=predictor.agent.desc(),
                                  reasoning=response.reasoning,
                                  confidence=response.confidence,
                                  signal=response.signal,
                                  analysis_details=response.analysis_details)
            agents_group.agents_list.append(result)
        return agents_group

    def train(self, date, future_data):
        agents_group = self.create_group(date)
        self.agent.handing_data(trade_date=date,
                                symbol=self.symbol,
                                agents_group=agents_group)
        long_prompt, mid_prompt, short_prompt, reflection_prompt = self.agent.query_records(
            trade_date=date, symbol=self.symbol)

        response = self.agent.generate_suggestion(
            date=date,
            symbol=self.symbol,
            short_prompt=short_prompt,
            mid_prompt=mid_prompt,
            long_prompt=long_prompt,
            reflection_prompt=reflection_prompt,
            returns=future_data['returns'].values[0])
        
        self.portfolio.update_market_info(
            cur_date=date,
            market_price=future_data['close'].values[0],
            rets=future_data['returns'].values[0])
        actions = 1 if future_data['returns'].values[0] > 0 else -1
        self.portfolio.record_action(action={'direction': actions})
        feedback = self.portfolio.feedback()
        self.agent.update_memory(trade_date=date,
                                    symbol=self.symbol,
                                    response=response,
                                    feedback=feedback)
        self.agent.save_checkpoint(path=os.path.join(os.environ['BASE_PATH'],
                                                'memory', self.agent.name,
                                                f'{self.symbol}_{date}'),
                              force=True)
        return response
