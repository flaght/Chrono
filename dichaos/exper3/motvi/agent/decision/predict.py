import pdb
from motvi.agent.predictor import Predictor as BasePredictor
from motvi.agent.decision.model import *
from motvi.agent.decision.agent import Agent


class Predictor(BasePredictor):

    def __init__(self, symbol, base_path, date):
        super(Predictor, self).__init__(symbol=symbol, agent_class=Agent)
        self.initialize_agent(base_path=base_path, symbol=symbol, date=date)
        self.agent_pool = []

    def add_sub_agent(self, agent: BasePredictor):
        self.agent_pool.append(agent)

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
    
    def predict(self, date, future_data=None):
        agents_group = self.create_group(date)
        if not agents_group:
            print(f"Skipping date {date} due to missing agent data.")
            return
        
        self.agent.handing_data(trade_date=date,
                                symbol=self.symbol,
                                agents_group=agents_group)
        long_prompt, mid_prompt, short_prompt, reflection_prompt = self.agent.query_records(
            trade_date=date, symbol=self.symbol)
        
        response = self.agent.generate_prediction(
            date=date,
            symbol=self.symbol,
            long_prompt=long_prompt,
            mid_prompt=mid_prompt,
            short_prompt=short_prompt,
            reflection_prompt=reflection_prompt)
        return response