from typing import Dict
from pydantic import BaseModel
import pdb, json, time
import pandas as pd
from dichaos.kdutils.logger import logger
from dichaos.agents.agents import Agents
from factors.calculate import *
from factors.model import *
from obility.model import create_suggestion_dom, create_prediction_dom
from .prompt import system_message, suggestion_human_message, decision_human_message

class Agent(Agents):
    name = 'obility'
    category = 'violent'

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
                     indicator_list: BaseModel, kline: BaseModel):
        memory = Memory(date=trade_date,
                        symbol=symbol,
                        indicator=indicator_list,
                        kline=kline,
                        index="-1")
        text = memory.model_dump_json()

        self.brain_db.add_memory_short_term(symbol=symbol,
                                            date=trade_date,
                                            text=text)
        
    def query_records(self, trade_date: str, symbol: str):

        def create_whole_prompts1(whole_data):
            str1 = ""
            for k, v in zip(whole_data[0], whole_data[1]):
                str1 += "过去反思记忆索引ID:{0}  内容:{1}\n".format("R" + str(v), k)
            return str1

        short_records = self.brain_db.query_memory_short_term(
            query_text="date is {0}".format(trade_date),
            symbol=symbol,
            top_k=self.top_k * 1000,
            duplicates=True)

        short_prompt = ""
        for index, record in zip(short_records[1], short_records[0]):
            memory = json.loads(record)
            memory = Memory(**memory)
            if memory.date != trade_date:
                continue
            memory.index = index
            short_prompt += memory.format(types='short')

        reflection_records = self.brain_db.query_memory_reflection(
            query_text="{0}".format(trade_date),
            symbol=symbol,
            top_k=int(self.top_k * 1.5),
            duplicates=False)
        reflection_prompt = create_whole_prompts1(reflection_records)
        return short_prompt, reflection_prompt
    
    def generate_suggestion(self, date: str, symbol: str, short_prompt: str,
                            reflection_prompt: str, chg: float):
        DomInfo1 = create_suggestion_dom(short_prompt=short_prompt,
                                         mid_prompt="",
                                         long_prompt="",
                                         reflection_prompt=reflection_prompt)
        #chg = round(future_data['ret_o2o'], 4)
        signal = "平盘" if abs(chg) < 0.00001 else ("上涨" if chg > 0 else "下跌")
        json_format = DomInfo1.dumps()
        for _ in range(5):
            response = self.generate_message(suggestion_human_message,
                                             params={
                                                 "ticker": symbol,
                                                 "date": date,
                                                 "chg": chg,
                                                 "signal": signal,
                                                 "short_terms": short_prompt,
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
                            reflection_prompt: str):

        DomInfo2 = create_prediction_dom(short_prompt=short_prompt,
                                         mid_prompt="",
                                         long_prompt="",
                                         reflection_prompt=reflection_prompt)

        json_format = DomInfo2.dumps()
        for _ in range(5):
            response = self.generate_message(decision_human_message,
                                             params={
                                                 "ticker": symbol,
                                                 "date": date,
                                                 "short_terms": short_prompt,
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
        if 'signal' in response.__dict__ and 'confidence' in response.__dict__:
            signal = response.signal
            confidence = response.confidence
            if confidence >= threshold and signal == 'bearish':
                return -1
            elif confidence >= threshold and signal == 'bullish':
                return 1
        return 0