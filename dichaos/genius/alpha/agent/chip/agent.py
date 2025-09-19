from typing import Dict
from agent.agents import Agents
from .model import *
from .prompt import *


class Agent(Agents):
    name = 'chip'
    category = 'alpha'

    def __init__(self, name: str, top_k: int, vector_provider: str,
                 db_name: str, embedding_model: str, embedding_provider: str,
                 llm_model: str, llm_provider: str, memory_params: Dict):
        super(Agent, self).__init__(name=name,
                                    top_k=top_k,
                                    vector_provider=vector_provider,
                                    db_name=db_name,
                                    embedding_model=embedding_model,
                                    embedding_provider=embedding_provider,
                                    llm_model=llm_model,
                                    llm_provider=llm_provider,
                                    memory_params=memory_params,
                                    system_message=system_message)

    def handing_data(self, trade_date: str, symbol: str,
                     factors_group: FactorsGroup):
        super(Agent, self).handing_data(trade_date=trade_date,
                                        symbol=symbol,
                                        factors_group=factors_group)

    def query_reflection(self, trade_date: str, symbol: str):
        return super(Agent, self).query_reflection(trade_date=trade_date,
                                                   symbol=symbol)

    def query_records(self, trade_date: str, symbol: str):
        return super(Agent, self).query_records(trade_date=trade_date,
                                                symbol=symbol)

    def generate_suggestion(self, date: str, symbol: str, short_prompt: str,
                            mid_prompt: str, long_prompt: str,
                            reflection_prompt: str, factors_details: str,
                            returns: float):
        return super(Agent, self).generate_suggestion(
            date=date,
            symbol=symbol,
            short_prompt=short_prompt,
            mid_prompt=mid_prompt,
            long_prompt=long_prompt,
            reflection_prompt=reflection_prompt,
            factors_details=factors_details,
            returns=returns,
            suggestion_human_message=suggestion_human_message)

    def generate_prediction(self, date: str, symbol: str, short_prompt: str,
                            mid_prompt: str, long_prompt: str,
                            reflection_prompt: str, factors_details: str):
        return super(Agent, self).generate_prediction(
            date=date,
            symbol=symbol,
            short_prompt=short_prompt,
            mid_prompt=mid_prompt,
            long_prompt=long_prompt,
            reflection_prompt=reflection_prompt,
            factors_details=factors_details,
            decision_human_message=decision_human_message)

    async def agenerate_suggestion(self, date: str, symbol: str,
                                   short_prompt: str, mid_prompt: str,
                                   long_prompt: str, reflection_prompt: str,
                                   factors_details: str, returns: float):
        # 使用 await 来调用父类的异步方法
        return await super(Agent, self).agenerate_suggestion(
            date=date,
            symbol=symbol,
            short_prompt=short_prompt,
            mid_prompt=mid_prompt,
            long_prompt=long_prompt,
            reflection_prompt=reflection_prompt,
            factors_details=factors_details,
            returns=returns,
            suggestion_human_message=suggestion_human_message)

    async def agenerate_prediction(self, date: str, symbol: str,
                                   short_prompt: str, mid_prompt: str,
                                   long_prompt: str, reflection_prompt: str,
                                   factors_details: str):
        return await super(Agent, self).agenerate_prediction(
            date=date,
            symbol=symbol,
            short_prompt=short_prompt,
            mid_prompt=mid_prompt,
            long_prompt=long_prompt,
            reflection_prompt=reflection_prompt,
            factors_details=factors_details,
            decision_human_message=decision_human_message)

    async def agenerate_prediction(self, date: str, symbol: str,
                                   short_prompt: str, mid_prompt: str,
                                   long_prompt: str, reflection_prompt: str,
                                   factors_details: str):
        return await super(Agent, self).agenerate_prediction(
            date=date,
            symbol=symbol,
            short_prompt=short_prompt,
            mid_prompt=mid_prompt,
            long_prompt=long_prompt,
            reflection_prompt=reflection_prompt,
            factors_details=factors_details,
            decision_human_message=decision_human_message)
    
    def actions(self, response, threshold=80):
        return super(Agent, self).actions(response=response,
                                          threshold=threshold)

    def desc(self):
        return desc
