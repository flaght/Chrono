import json, time, pdb
from typing import Dict
from pydantic import BaseModel
from dichaos.kdutils.logger import logger
from dichaos.agents.agents import Agents
from .model import *
from .prompt import system_message, suggestion_human_message
from motvi.kdutils.model import *


class Agent(Agents):
    name = 'posflow'
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
                     factors_group: FactorsGroup):
        self.brain_db.add_memory_short_term(
            symbol=symbol,
            date=trade_date,
            text=factors_group.model_dump_json())

    def query_record(self, trade_date: str, symbol: str, name: str):
        query_memory_term = getattr(self.brain_db,
                                    "query_memory_{0}_term".format(name))
        short_records = query_memory_term(
            query_text="date must is {0}".format(trade_date),
            symbol=symbol,
            top_k=self.top_k * 1000,
            duplicates=True)

        prompt = ""
        for index, record in zip(short_records[1], short_records[0]):
            memory = json.loads(record)
            memory = FactorsGroup(**memory)
            if memory.date != trade_date:
                continue
            memory.index = index
            prompt += memory.format(types=name)
        return prompt

    def query_reflection(self, trade_date: str, symbol: str):

        def create_whole_prompts1(whole_data):
            str1 = ""
            for k, v in zip(whole_data[0], whole_data[1]):
                str1 += "过去反思记忆索引ID:{0}  内容:{1}\n\n".format("R" + str(v), k)
            return str1

        reflection_records = self.brain_db.query_memory_reflection(
            query_text="{0}".format(trade_date),
            symbol=symbol,
            top_k=int(self.top_k * 1.5),
            duplicates=False)
        reflection_prompt = create_whole_prompts1(reflection_records)
        return reflection_prompt

    def query_records(self, trade_date: str, symbol: str):
        short_prompt = self.query_record(trade_date=trade_date,
                                         symbol=symbol,
                                         name='short')
        mid_prompt = self.query_record(trade_date=trade_date,
                                       symbol=symbol,
                                       name='mid')
        long_prompt = self.query_record(trade_date=trade_date,
                                        symbol=symbol,
                                        name='long')

        reflection_prompt = self.query_reflection(trade_date=trade_date,
                                                  symbol=symbol)

        return long_prompt, mid_prompt, short_prompt, reflection_prompt

    def generate_suggestion(self, date: str, symbol: str, short_prompt: str,
                            mid_prompt: str, long_prompt: str,
                            reflection_prompt: str, factors_details: str,
                            returns: float):
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
                                                 "factors_details":
                                                 factors_details,
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

    def update_memory(self, trade_date: str, symbol: str, response: any,
                      feedback: dict):
        super(Agent, self).update_memory(trade_date=trade_date,
                                         symbol=symbol,
                                         response=response,
                                         feedback=feedback)
