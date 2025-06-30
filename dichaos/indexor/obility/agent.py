from typing import Dict
from pydantic import BaseModel
import pdb, json, time
import pandas as pd
from dichaos.kdutils.logger import logger
from dichaos.agents.indexor.ability.model import Memory
from dichaos.agents.agents import Agents
from .calculate import *
from .model import *
from .prompt import system_message, suggestion_human_message, decision_human_message


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

    def create_kline(self, date, symbol, open, close, high, low, volume):
        return KLine(date=date,
                     symbol=symbol,
                     open=open,
                     close=close,
                     high=high,
                     low=low,
                     volume=volume)

    def create_indicator(self, date, sma5, sma10, sma20, ema12, ema26, rsi,
                         macd, atr, vwap, adx, obv, pp, r1, s1, r2, s2, r3,
                         s3):
        indicator_list = IndicatorList(date=date)
        indicator_list.set_indicator(sma5=sma5,
                                     sma10=sma10,
                                     sma20=sma20,
                                     ema12=ema12,
                                     ema26=ema26,
                                     rsi=rsi,
                                     macd=macd,
                                     atr=atr,
                                     vwap=vwap,
                                     adx=adx,
                                     obv=obv,
                                     pp=pp,
                                     r1=r1,
                                     s1=s1,
                                     r2=r2,
                                     s2=s2,
                                     r3=r3,
                                     s3=s3)
        return indicator_list

    ### 数据集放入进去 所有的指标算一遍，回测的话 返回批量周期因子，实盘的返回最新因子
    def calcuate_sma(self, data: pd.DataFrame):
        """
        Calculate the Simple Moving Average (SMA) for a given period.
        """
        sma5 = calcuate_sma(data, period=5)
        sma10 = calcuate_sma(data, period=10)
        sma20 = calcuate_sma(data, period=20)
        return sma5, sma10, sma20

    def calculate_ema(self, data: pd.DataFrame):
        """
        Calculate the Exponential Moving Average (EMA) for a given period.
        """
        ema12 = calculate_ema(data, period=12)
        ema26 = calculate_ema(data, period=26)
        return ema12, ema26

    def calculate_rsi(self, data: pd.DataFrame):
        """
        Calculate the Relative Strength Index (RSI) for a given period.
        """
        return calculate_rsi(data, period=14)

    def calculate_macd(self, data: pd.DataFrame):
        """
        Calculate the Moving Average Convergence Divergence (MACD) for a given period.
        """
        return calculate_macd(data,
                              fast_period=12,
                              slow_period=26,
                              signal_period=9)

    def calculate_bollinger_bands(self, data: pd.DataFrame):
        """
        Calculate the Bollinger Bands for a given period.
        """
        return calculate_bollinger_bands(data, period=20, std_dev=2)

    def calculate_atr(self, data: pd.DataFrame):
        """
        Calculate the Average True Range (ATR) for a given period.
        """
        return calculate_atr(data, period=14)

    def calculate_vwap(self, data: pd.DataFrame):
        """
        Calculate the Volume Weighted Average Price (VWAP) for a given period.
        """
        return calculate_vwap(data)

    def calculate_adx(self, data: pd.DataFrame):
        """
        Calculate the Average Directional Index (ADX) for a given period.
        """
        return calculate_adx(data, period=14)

    def calculate_obv(self, data: pd.DataFrame):
        """
        Calculate the On-Balance Volume (OBV) for a given period.
        """
        return calculate_obv(data)

    def calcuate_point(self, data: pd.DataFrame):
        """
        Calculate the point for a given period.
        """
        point = calcuate_point(data)
        return point

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

    def update_memory(self, trade_date: str, symbol: str, response: any,
                      feedback: dict):
        #feed = 1 if feedback[trade_date] > 0 else -1
        #feed = {'feedback': feed, 'date': trade_date}
        feed = feedback
        memory = response.model_dump()
        if 'short_memory_index' in memory:
            index_array = memory['short_memory_index'].split(',')
            if len(index_array) > 0:
                memory['short_memory_index'] = [{
                    'memory_index':
                    int(index.split('S')[-1])
                } for index in index_array]

        if 'mid_memory_index' in memory:
            index_array = memory['mid_memory_index'].split(',')
            memory['mid_memory_index'] = [{
                'memory_index':
                int(index.split('M')[-1])
            } for index in index_array]

        if 'long_memory_index' in memory:
            index_array = memory['long_memory_index'].split(',')
            memory['long_memory_index'] = [{
                'memory_index':
                int(index.split('L')[-1])
            } for index in index_array]

        if 'reflection_memory_index' in memory:
            index_array = memory['reflection_memory_index'].split(',')
            memory['reflection_memory_index'] = [{
                'memory_index':
                int(index.split('R')[-1])
            } for index in index_array]

        if 'summary_reason' in memory:
            self.brain_db.add_memory_reflection(symbol=symbol,
                                                date=trade_date,
                                                text=memory['summary_reason'])
        if 'short_memory_index' in memory:
            self.brain_db.update_access_count(
                symbol=symbol,
                cur_memory=memory,
                layer_index_name='short_memory_index',
                feedback=feed)
        if 'mid_memory_index' in memory:
            self.brain_db.update_access_count(
                symbol=symbol,
                cur_memory=memory,
                layer_index_name='mid_memory_index',
                feedback=feed)

        if 'long_memory_index' in memory:
            self.brain_db.update_access_count(
                symbol=symbol,
                cur_memory=memory,
                layer_index_name='long_memory_index',
                feedback=feed)

        if 'reflection_memory_index' in memory:
            self.brain_db.update_access_count(
                symbol=symbol,
                cur_memory=memory,
                layer_index_name='reflection_memory_index',
                feedback=feed)

        self.brain_db.step()

    def generate_suggestion(self, date: str, symbol: str, short_prompt: str,
                            reflection_prompt: str, future_data: dict):
        DomInfo1 = create_suggestion_dom(short_prompt=short_prompt,
                                         mid_prompt="",
                                         long_prompt="",
                                         reflection_prompt=reflection_prompt)
        chg = round(future_data['ret_o2o'], 4)
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
