import json, time, pdb
from typing import Dict
from pydantic import BaseModel
from dichaos.kdutils.logger import logger
from .model import *
from .prompt import system_message, suggestion_human_message, decision_human_message
from motvi.kdutils.model import *
from motvi.agent.agents import Agents


class Agent(Agents):
    name = 'decision'
    category = 'motvi'

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
                     agents_group: AgentsGroup):
        super(Agent, self).handing_data(trade_date=trade_date,
                                        symbol=symbol,
                                        factors_group=agents_group)

    def query_reflection(self, trade_date: str, symbol: str):
        return super(Agent, self).query_reflection(trade_date=trade_date,
                                                   symbol=symbol)

    def query_record(self, trade_date: str, symbol: str, name: str):
        query_memory_term = getattr(self.brain_db,
                                    "query_memory_{0}_term".format(name))
        records = query_memory_term(
            query_text="date must is {0}".format(trade_date),
            symbol=symbol,
            top_k=self.top_k * 1000,
            duplicates=True)

        prompt = ""
        for index, record in zip(records[1], records[0]):
            memory = json.loads(record)
            memory = AgentsGroup(**memory)
            if memory.date != trade_date:
                continue
            memory.index = index
            prompt += memory.format(types=name)
        return prompt

    def query_records(self, trade_date: str, symbol: str):
        return super(Agent, self).query_records(trade_date=trade_date,
                                                symbol=symbol)

    def generate_suggestion(self, date: str, symbol: str, short_prompt: str,
                            mid_prompt: str, long_prompt: str,
                            reflection_prompt: str, returns: float):
        DomInfo1 = create_suggestion_dom(short_prompt=short_prompt,
                                         mid_prompt=mid_prompt,
                                         long_prompt=long_prompt,
                                         reflection_prompt=reflection_prompt)
        signal = "平盘" if abs(returns) < 0.00001 else (
            "上涨" if returns > 0 else "下跌")
        json_format = DomInfo1.dumps()
        for _ in range(5):
            response = self.generate_message(suggestion_human_message,
                                             params={
                                                 "ticker": symbol,
                                                 "date": date,
                                                 "chg": round(returns, 4),
                                                 "signal": signal,
                                                 "short_terms": short_prompt,
                                                 "mid_terms": mid_prompt,
                                                 "long_terms": long_prompt,
                                                 "reflection_terms":
                                                 reflection_prompt,
                                                 "json_format": json_format
                                             },
                                             default={},
                                             response=DomInfo1,
                                             is_structured=True)
            try:
                if 'summary_reason' in response.__dict__:
                    break
                else:
                    logger.info('retrying...')
                    time.sleep(5)
                    continue
            except Exception as e:
                logger.info('error:{0}'.format(e))
                time.sleep(5)
                continue
        return response

    def generate_prediction(self, date: str, symbol: str, short_prompt: str,
                            mid_prompt: str, long_prompt: str,
                            reflection_prompt: str):
        DomInfo2 = create_prediction_dom1(short_prompt=short_prompt,
                                         mid_prompt=mid_prompt,
                                         long_prompt=long_prompt,
                                         reflection_prompt=reflection_prompt)

        json_format = DomInfo2.dumps()
        for _ in range(5):
            response = self.generate_message(decision_human_message,
                                             params={
                                                 "ticker": symbol,
                                                 "date": date,
                                                 "short_terms": short_prompt,
                                                 "mid_terms": mid_prompt,
                                                 "long_terms": long_prompt,
                                                 "reflection_terms":
                                                 reflection_prompt,
                                                 "json_format": json_format
                                             },
                                             default={},
                                             response=DomInfo2,
                                             is_structured=True)
            try:
                if 'reasoning' in response.__dict__:
                    break
                else:
                    logger.info('retrying...')
                    time.sleep(5)
                    continue
            except Exception as e:
                logger.info('error:{0}'.format(e))
                time.sleep(5)
                continue
        return response

    def actions(self, response, threshold=80):
        return super(Agent, self).actions(response=response,
                                          threshold=threshold)
