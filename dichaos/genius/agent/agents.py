import json, time, pdb, asyncio
from typing import Dict
from pydantic import BaseModel
from dichaos.kdutils.logger import logger
from dichaos.agents.agents import Agents as BaseAgents
from kdutils.model import *


class Agents(BaseAgents):
    name = 'base'
    category = 'genius'

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

    def update_memory(self, trade_date: str, symbol: str, response: any,
                      feedback: dict):
        super(Agents, self).update_memory(trade_date=trade_date,
                                          symbol=symbol,
                                          response=response,
                                          feedback=feedback)

    def actions(self, response, threshold=80):
        if 'signal' in response.__dict__ and 'confidence' in response.__dict__:
            signal = response.signal
            confidence = response.confidence
            if confidence >= threshold and signal == 'bearish':
                return -1
            elif confidence >= threshold and signal == 'bullish':
                return 1
        return 0

    def generate_suggestion(self, date: str, symbol: str, short_prompt: str,
                            mid_prompt: str, long_prompt: str,
                            reflection_prompt: str, factors_details: str,
                            returns: float, suggestion_human_message: str):
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

    def generate_prediction(self, date: str, symbol: str, short_prompt: str,
                            mid_prompt: str, long_prompt: str,
                            reflection_prompt: str, factors_details: str,
                            decision_human_message: str):

        DomInfo2 = create_prediction_dom(short_prompt=short_prompt,
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
                                                 "json_format": json_format,
                                                 "factors_details":
                                                 factors_details
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

    async def agenerate_prediction(self, date: str, symbol: str,
                                   short_prompt: str, mid_prompt: str,
                                   long_prompt: str, reflection_prompt: str,
                                   factors_details: str,
                                   decision_human_message: str):
        DomInfo2 = create_prediction_dom(short_prompt=short_prompt,
                                         mid_prompt=mid_prompt,
                                         long_prompt=long_prompt,
                                         reflection_prompt=reflection_prompt)
        json_format = DomInfo2.dumps()
        for i in range(5):
            try:
                response = await self.agenerate_message(
                    decision_human_message,
                    params={
                        "ticker": symbol,
                        "date": date,
                        "short_terms": short_prompt,
                        "mid_terms": mid_prompt,
                        "long_terms": long_prompt,
                        "reflection_terms": reflection_prompt,
                        "json_format": json_format,
                        "factors_details": factors_details
                    },
                    default={},
                    response_schema=DomInfo2)

                # 检查响应是否有效
                if response and hasattr(
                        response,
                        '__dict__') and 'reasoning' in response.__dict__:
                    break  # 成功，跳出循环
                else:
                    logger.info(f'Retrying... (Attempt {i+1}/5)')
                    await asyncio.sleep(5)  # 使用异步休眠，不会阻塞事件循环
            except Exception as e:
                logger.info(f'Error on attempt {i+1}/5: {e}')
                await asyncio.sleep(5)  # 发生错误时也使用异步休眠

        return response

    async def agenerate_suggestion(self, date: str, symbol: str,
                                   short_prompt: str, mid_prompt: str,
                                   long_prompt: str, reflection_prompt: str,
                                   factors_details: str, returns: float,
                                   suggestion_human_message: str):
        DomInfo1 = create_suggestion_dom(short_prompt=short_prompt,
                                         mid_prompt=mid_prompt,
                                         long_prompt=long_prompt,
                                         reflection_prompt=reflection_prompt)
        signal = "平盘" if abs(returns) < 0.00001 else (
            "上涨" if returns > 0 else "下跌")
        json_format = DomInfo1.dumps()

        response = None  # 初始化 response
        for i in range(5):
            try:
                response = await self.agenerate_message(
                    suggestion_human_message,
                    params={
                        "ticker": symbol,
                        "date": date,
                        "chg": round(returns, 4),
                        "signal": signal,
                        "short_terms": short_prompt,
                        "mid_terms": mid_prompt,
                        "long_terms": long_prompt,
                        "reflection_terms": reflection_prompt,
                        "factors_details": factors_details,
                        "json_format": json_format
                    },
                    default={},
                    response_schema=DomInfo1  #,
                    #is_structured=True
                )

                # 检查响应是否有效
                if response and hasattr(
                        response,
                        '__dict__') and 'summary_reason' in response.__dict__:
                    break  # 成功，跳出循环
                else:
                    logger.info(f'Retrying... (Attempt {i+1}/5)')
                    await asyncio.sleep(5)  # 使用异步休眠，不会阻塞事件循环

            except Exception as e:
                logger.info(f'Error on attempt {i+1}/5: {e}')
                await asyncio.sleep(5)  # 发生错误时也使用异步休眠

        return response
