import pdb, json, time
from typing import Dict, List
from pydantic import BaseModel
from dichaos.kdutils.logger import logger
from dichaos.agents.agents import Agents
from .prompt import system_message, suggestion_human_message
from obility.model import *
from langchain.output_parsers import PydanticOutputParser


class Agent(Agents):
    name = 'obility'
    category = 'indexor'

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

    def handing_filling(self, trade_date: str, code: str, indicator: BaseModel,
                        kline: BaseModel):
        ### 遍历存储
        self.brain_db.add_memory_short_term(symbol=code,
                                            date=trade_date,
                                            text=json.dumps({
                                                'indicator':
                                                indicator.model_dump(),
                                                'kline':
                                                kline.model_dump()
                                            }))

    def query_memory(self, trade_date: str, codes: List):
        short_memory = {}
        reflection_memory = {}
        for code in codes:
            ## 查询短期
            fillings = self.brain_db.query_memory_short_term(
                query_text="find {0}  data".format(trade_date),
                top_k=5,
                symbol=code)

            for index, record in zip(fillings[1], fillings[0]):
                filling = json.loads(record)
                indicator = IndicatorSets(**filling['indicator'])
                kline = KLine(**filling['kline'])
                indicator_list = IndicatorList(code=indicator.code,
                                               date=indicator.date,
                                               indicator=indicator,
                                               kline=kline)
                short_memory[indicator.code] = indicator_list.to_json(
                    index="S{0}".format(index))

            ## 查询反思
            fillings = self.brain_db.query_memory_reflection(
                query_text="{0}".format(trade_date),
                symbol=code,
                top_k=int(self.top_k * 1.5),
                duplicates=False)

            memory = []
            for index, record in zip(fillings[1], fillings[0]):
                memory.append({"R{0}".format(index): record})
            if len(memory) > 0:
                reflection_memory[code] = memory

        ### 整体经验
        fillings = self.brain_db.query_memory_reflection(
            query_text="{0}".format(trade_date),
            symbol=self.name,
            top_k=int(self.top_k * 1.5),
            duplicates=False)

        memory = []
        for index, record in zip(fillings[1], fillings[0]):
            memory.append({"R{0}".format(index): record})
        if len(memory) > 0:
            reflection_memory['details'] = memory

        return short_memory, reflection_memory

    def generate_suggestion(self, date: str, short_prompt: str,
                            reflection_prompt: str, overview_memory: str):
        DomInfos1 = create_suggestion_dom(short_prompt=short_prompt,
                                          mid_prompt="",
                                          long_prompt="",
                                          reflection_prompt=reflection_prompt)

        for _ in range(5):
            response = self.generate_message(suggestion_human_message,
                                             params={
                                                 "short_terms":
                                                 short_prompt,
                                                 "reflection_terms":
                                                 reflection_prompt,
                                                 "maarket_overview":
                                                 overview_memory
                                             },
                                             default={},
                                             response=DomInfos1,
                                             is_structured=True)
            try:
                if 'details' in response.__dict__:
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

    def update_memory(self, trade_date: str, response: any, feedback: dict):
        for resp in response.tickers:
            if resp.code not in feedback:
                continue
            fb = feedback[resp.code]
            super(Agent, self).update_memory(trade_date=trade_date,
                                             symbol=resp.code,
                                             response=resp,
                                             feedback=fb)

        ## 整体detail 更新
        self.brain_db.add_memory_reflection(
            symbol=self.name,  ## 用agent名
            date=trade_date,
            text=response.details)
