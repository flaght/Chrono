from typing import Dict
from pydantic import BaseModel
import pdb, json, time
import pandas as pd
from dichaos.kdutils.logger import logger
from dichaos.agents.agents import Agents
from .prompt import system_message, suggestion_human_message, decision_human_message
from kdutils.until import create_agent_path
from decision.model import TraderSignal, Memory, create_suggestion_dom, create_prediction_dom


class Agent(Agents):
    name = 'decision'
    category = 'indexor'

    def __init__(self, name: str, top_k: int, vector_provider: str,
                 db_name: str, embedding_model: str, embedding_provider: str,
                 llm_model: str, llm_provider: str, memory_params: Dict,
                 **kwargs):
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
        self.signal_agents = []

    def set_agents(self, agent):
        self.signal_agents.append(agent)

    def calcuate_sma(self, total_data: pd.DataFrame):
        return self.signal_agents[0].calcuate_sma(total_data)

    def calculate_ema(self, total_data: pd.DataFrame):
        return self.signal_agents[0].calculate_ema(total_data)

    def calculate_rsi(self, total_data: pd.DataFrame):
        return self.signal_agents[0].calculate_rsi(total_data)

    def calculate_macd(self, total_data: pd.DataFrame):
        return self.signal_agents[0].calculate_macd(total_data)

    def calculate_bollinger_bands(self, total_data: pd.DataFrame):
        return self.signal_agents[0].calculate_bollinger_bands(total_data)

    def calculate_atr(self, total_data: pd.DataFrame):
        return self.signal_agents[0].calculate_atr(total_data)

    def calculate_vwap(self, total_data: pd.DataFrame):
        return self.signal_agents[0].calculate_vwap(total_data)

    def calculate_adx(self, total_data: pd.DataFrame):
        return self.signal_agents[0].calculate_adx(total_data)

    def calculate_obv(self, total_data: pd.DataFrame):
        return self.signal_agents[0].calculate_obv(total_data)

    def calcuate_point(self, total_data: pd.DataFrame):
        return self.signal_agents[0].calcuate_point(total_data)

    def create_kline(self, date, symbol, open, close, high, low, volume):
        return [
            agent.create_kline(date=date,
                               symbol=symbol,
                               open=open,
                               close=close,
                               high=high,
                               low=low,
                               volume=volume) for agent in self.signal_agents
        ]

    def create_indicator_list(self, date, sma5, sma10, sma20, ema12, ema26,
                              rsi, macd, atr, vwap, adx, obv, pp, r1, s1, r2,
                              s2, r3, s3):
        return [
            agent.create_indicator(date=date,
                                   sma5=sma5,
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
                                   s3=s3) for agent in self.signal_agents
        ]

    def handing_data(self, trade_date: str, symbol: str,
                     indicator_list: BaseModel, kline: BaseModel):
        response_sets = []
        for index, agent in enumerate(self.signal_agents):
            if not hasattr(agent, 'handing_data'):
                continue
            agent.handing_data(trade_date, symbol, indicator_list[index],
                               kline[index])
            short_prompt, reflection_prompt = agent.query_records(
                trade_date, symbol)

            response = agent.generate_prediction(
                date=trade_date,
                symbol=symbol,
                short_prompt=short_prompt,
                reflection_prompt=reflection_prompt)
            response_sets.append(
                TraderSignal(name=agent.name,
                             reasoning=response.reasoning,
                             confidence=response.confidence,
                             signal=response.signal,
                             analysis_details=response.analysis_details))
        texts = [response.format() for response in response_sets]
        memory = Memory(date=trade_date,
                        symbol=symbol,
                        response="\n\n".join(texts),
                        index="-1")
        self.brain_db.add_memory_short_term(symbol=symbol,
                                            date=trade_date,
                                            text=memory.model_dump_json())

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

        if 'summary_reason' in memory or 'analysis_details' in memory:
            reflection = {}
            if 'summary_reason' in memory:
                reflection['summary_reason'] = memory['summary_reason']
            if 'analysis_details' in memory:
                reflection['analysis_details'] = memory['analysis_details']
            text = "summary_reason:{0}\n analysis_details:{1}".format(
                reflection['summary_reason'], reflection['analysis_details'])

            self.brain_db.add_memory_reflection(symbol=symbol,
                                                date=trade_date,
                                                text=text)
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
                            reflection_prompt: str, risk_data: BaseModel,
                            portfolio: Dict):
        DomInfo2 = create_prediction_dom(short_prompt=short_prompt,
                                         mid_prompt="",
                                         long_prompt="",
                                         reflection_prompt=reflection_prompt)
        json_format = DomInfo2.dumps()
        for _ in range(5):
            response = self.generate_message(
                decision_human_message,
                params={
                    "ticker": symbol,
                    "date": date,
                    "short_terms": short_prompt,
                    "reflection_terms": reflection_prompt,
                    "current_price": risk_data.current_price,
                    "max_shares":
                    risk_data.remaining_limit / risk_data.current_price,
                    "portfolio_cash": risk_data.available_cash,
                    "portfolio_positions": portfolio.get('positions', {}),
                    "margin_requirement":
                    portfolio.get('margin_requirement', 0),
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
